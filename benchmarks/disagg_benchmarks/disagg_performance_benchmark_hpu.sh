#!/bin/bash

# Requirement: 8x Gaudi2 HPUs.


# Model: meta-llama/Meta-Llama-3.1-70B-Instruct 
# Query: 2048 input tokens, 11 output tokens, QPS 4, 500 requests
# Resource: 8x Gaudi2
# Approaches:
# no chunked prefill since habana vllm doesn't support it yet.
# 1. 2 vllm instance with tp=4, equivalent to 1 tp=4 instance with QPS 4
# 2. Disaggregated prefill: 1 prefilling instance and 1 decoding instance
# Prefilling instance: max_output_token=1
# Decoding instance: force the input tokens be the same across requests to bypass prefilling

set -ex

kill_hpu_processes() {
  # kill all processes on HPU.
  # pkill -f pt_main_thread
  # pkill -f python3
  ps -e | grep pt_main_thread | awk '{print $1}' | xargs kill -9
  ps -e | grep python3 | awk '{print $1}' | xargs kill -9
  for port in 8000 8100 8200; do lsof -t -i:$port | xargs -r kill -9; done
  sleep 1
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

# no chunked prefill since habana vllm doesn't support it yet.
launch_baseline_prefill() {
  model="shenzhi-wang/Llama3.1-8B-Chinese-Chat"  #"meta-llama/Meta-Llama-3.1-70B-Instruct"
  # disagg prefill
  HABANA_VISIBLE_MODULES="0" python3 \
      -m vllm.entrypoints.openai.api_server \
      --model $model \
      --port 8100 \
      -tp 1 \
      --max-model-len 10000 \
      --disable-log-stats \
      --disable-log-requests \
      --num-scheduler-steps 1 \
      --gpu-memory-utilization 0.8 &
  HABANA_VISIBLE_MODULES="1" python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8200 \
    -tp 1 \
    --max-model-len 10000 \
    --disable-log-stats \
    --disable-log-requests \
    --num-scheduler-steps 1 \
    --gpu-memory-utilization 0.8 &
  wait_for_server 8100
  wait_for_server 8200
  python3 round_robin_proxy.py &
  sleep 1
}


launch_disagg_prefill() {
  model="shenzhi-wang/Llama3.1-8B-Chinese-Chat"   #"meta-llama/Meta-Llama-3.1-70B-Instruct" 
  # disagg prefill
  # 17k-0.5
  # --max-seq-len-to-capture
  # --enforce-eager \
  VLLM_PORT=12345 VLLM_DISTRIBUTED_KV_ROLE=producer python3 \
      -m vllm.entrypoints.openai.api_server \
      --model $model \
      --port 8100 \
      -tp 1 \
      --max-model-len 10000 \
      --disable-log-stats \
      --disable-log-requests \
      --gpu-memory-utilization 0.8 &
  VLLM_PORT=12345 VLLM_DISTRIBUTED_KV_ROLE=consumer python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8200 \
    -tp 1 \
    --max-model-len 10000 \
    --disable-log-stats \
    --disable-log-requests \
    --gpu-memory-utilization 0.8 &
  wait_for_server 8100
  wait_for_server 8200
  python3 disagg_prefill_proxy_server.py &
  sleep 1
}


benchmark() {
  results_folder="./results"
  model="shenzhi-wang/Llama3.1-8B-Chinese-Chat"   #"meta-llama/Meta-Llama-3.1-70B-Instruct"
  dataset_name="sonnet"
  dataset_path="../sonnet_4x.txt"
  num_prompts=400
  qps=$1
  prefix_len=50
  input_len=2048
  output_len=$2
  tag=$3
  rag_bench=$4
  prepare_rag=$5

  args=()
  args+=(--backend vllm)
  args+=(--model $model)
  args+=(--dataset-name $dataset_name)
  args+=(--dataset-path $dataset_path)
  args+=(--sonnet-input-len $input_len)
  args+=(--sonnet-output-len $output_len)
  args+=(--sonnet-prefix-len $prefix_len)
  args+=(--num-prompts $num_prompts)
  args+=(--port 8000)
  args+=(--save-result)
  args+=(--result-dir $results_folder)
  args+=(--result-filename $tag-qps-$qps.json)
  args+=(--request-rate $qps)
  # avoiding empty first token if meets eos_token_id
  args+=(--ignore-eos)
  if [ "${rag_bench}" = "true" ]; then
    args+=(--rag_kv-cache-offload-bench)
  fi
  if [ "${prepare_rag}" = "true" ]; then
    args+=(--prepare-rag-kv-cache)
  fi
  echo "Running: python3 ../benchmark_serving.py ${args[@]}"

  python3 ../benchmark_serving.py ${args[@]}

  sleep 2

}


main() {

  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get update && apt-get -y install jq)
  (which socat) || (apt-get update && apt-get -y install socat)

  pip install quart httpx matplotlib aiohttp

  cd "$(dirname "$0")"

  cd ..
  # create sonnet-4x.txt so that we can sample 2048 tokens for input
  echo "" > sonnet_4x.txt
  for _ in {1..4}
  do
    cat sonnet.txt >> sonnet_4x.txt
  done
  cd disagg_benchmarks

  rm -rf results
  mkdir results

  default_output_len=32

  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
  # "simple_buffer"
  export VLLM_KV_TRANSFER_DRIVER="disk_kv_transfer"

  # required to be true for HCCL collectives with HPU Graphs
  # export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
  # export PT_HPU_LAZY_MODE=0
  # for hpu graphs warmup
  # export VLLM_PROMPT_BS_BUCKET_MIN=1      # (default:1)
  # export VLLM_PROMPT_BS_BUCKET_STEP=4     # (default:32)
  # export VLLM_PROMPT_BS_BUCKET_MAX=32     # (default:64)

  # # input_len - 2*prefix_len ~ input_len + 2*prefix_len
  # # divisible by block_size 128
  # export VLLM_PROMPT_SEQ_BUCKET_MIN=896   # (default:128)
  # export VLLM_PROMPT_SEQ_BUCKET_STEP=128  # (default:128)
  # export VLLM_PROMPT_SEQ_BUCKET_MAX=1152  # (default:1024)

  # export VLLM_DECODE_BS_BUCKET_MIN=1      # (default:1)
  # export VLLM_DECODE_BS_BUCKET_STEP=8     # (default:32)
  # export VLLM_DECODE_BS_BUCKET_MAX=64     # (default:256)

  # prepare rag kv cache
  # launch_disagg_prefill
  # for qps in 4; do
  # benchmark $qps 2 save_rag true true
  # done
  # kill_hpu_processes

  launch_baseline_prefill
  for qps in 2 4 6 8; do
  benchmark $qps $default_output_len baseline_prefill
  done
  kill_hpu_processes

  launch_disagg_prefill
  for qps in 2 4 6 8; do
  benchmark $qps $default_output_len disagg_prefill true false
  done
  kill_hpu_processes

  python3 visualize_benchmark_results.py --device "hpu"

}


main "$@"
