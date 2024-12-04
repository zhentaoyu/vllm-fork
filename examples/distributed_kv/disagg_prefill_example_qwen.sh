#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling
# We will launch 2 vllm instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export VLLM_PORT=12345
model="mQwen/Qwen2-7B-Instruct"

# install quart first -- required for disagg prefill proxy serve
if python3 -c "import quart" &> /dev/null; then
    echo "Quart is already installed."
else
    echo "Quart is not installed. Installing..."
    python3 -m pip install quart
    python3 -m pip install flask==3.0.3
fi 

# a function that waits vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

export VLLM_KV_TRANSFER_DRIVER="disk_kv_transfer"

# prefilling instance, which is the KV producer
VLLM_DISTRIBUTED_KV_ROLE=producer CUDA_VISIBLE_DEVICES=0 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model mQwen/Qwen2-7B-Instruct \
    --port 8100 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.8 &

# decoding instance, which is the KV consumer
VLLM_DISTRIBUTED_KV_ROLE=consumer CUDA_VISIBLE_DEVICES=1 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model mQwen/Qwen2-7B-Instruct \
    --port 8200 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.8 &

# wait until prefill and decode instances are ready
wait_for_server 8100
wait_for_server 8200

# launch a proxy server that opens the service at port 8000
# the workflow of this proxy:
# - send the request to prefill vLLM instance (port 8100), change max_tokens to 1
# - after the prefill vLLM finishes prefill, send the request to decode vLLM instance
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY_SERVER_SCRIPT="$SCRIPT_DIR/../../benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py"
python3 ${PROXY_SERVER_SCRIPT} &
sleep 1

# serve two example requests
output1=$(curl -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "mQwen/Qwen2-7B-Instruct",
"prompt": "San Francisco is a",
"max_tokens": 10,
"temperature": 0
}')

output2=$(curl -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "mQwen/Qwen2-7B-Instruct",
"prompt": "Santa Clara is a",
"max_tokens": 10,
"temperature": 0
}')

output3=$(curl -s http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d @- <<EOF
{
"model": "mQwen/Qwen2-7B-Instruct",
"messages": [{"role": "user", "content": "Tell me something about Intel"}],
"max_completion_tokens": 10,
"temperature": 0
}
EOF
)

# Print the outputs of the curl requests
echo ""
echo "Output of first request: $output1"
echo "Output of second request: $output2"
echo "Output of third request: $output3"

echo "Successfully finished 3 test requests!"
echo ""

tail -f /dev/null
# Cleanup commands, suppressing their output
# ps -e | grep pt_main_thread | awk '{print $1}' | xargs kill -9 > /dev/null 2>&1
# pkill -f python3 > /dev/null 2>&1
