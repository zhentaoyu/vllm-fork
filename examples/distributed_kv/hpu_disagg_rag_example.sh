#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling with rag.
# We will launch 2 vllm instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.
# test steps:
#
# 1. send system_prompt + rag and save its kv cache into disk
# a template is like below:
# ### You are a helpful, respectful and honest assistant to help the user with questions. \
# Please refer to the search results obtained from the local knowledge base. \
# But be careful to not incorporate the information that you think is not relevant to the question. \
# If you don't know the answer to a question, please don't share false information. \n
# ### Search results: {context} \n
# ### Question: {question} \n
# ### Answer:
#
# 2. send system_prompt + rag + question and let prefill load knowledge kv cache and continue
#    prefill question
#
# 3. decode ins load all kv caches and preform decoding

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export VLLM_PORT=12345

# install quart first -- required for disagg prefill proxy serve
if python3 -c "import quart" &> /dev/null; then
    echo "Quart is already installed."
else
    echo "Quart is not installed. Installing..."
    python3 -m pip install quart
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
# no FUSESDPA in pre-save kv cache period since prompt_attention_with_context has no FUSESDPA either
# it seems would affect accuracy if turn FUSESDPA on
export VLLM_PROMPT_USE_FUSEDSDPA=0
export VLLM_CONTIGUOUS_PA=false

# prefilling instance, which is the KV producer
 VLLM_DISTRIBUTED_KV_ROLE=producer python3 \
    -m vllm.entrypoints.openai.api_server \
    --model shenzhi-wang/Llama3.1-8B-Chinese-Chat \
    --port 8100 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.8 &

# decoding instance, which is the KV consumer
VLLM_DISTRIBUTED_KV_ROLE=consumer python3 \
    -m vllm.entrypoints.openai.api_server \
    --model shenzhi-wang/Llama3.1-8B-Chinese-Chat \
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

system_prompt="### You are a helpful, respectful and honest assistant to help the user with questions. \
Please refer to the search results obtained from the local knowledge base. \
But be careful to not incorporate the information that you think is not relevant to the question. \
If you don't know the answer to a question, please don't share false information. \n"

rag_prefix="### Search results:"

rag=" In a world where technology has advanced beyond our wildest dreams, humanity stands \
on the brink of a new era. The year is 2050, and artificial intelligence has become an integral part of \
everyday life. Autonomous vehicles zip through the bustling streets, drones deliver packages with pinpoint \
accuracy, and smart homes anticipate every need of their inhabitants with ease. But with these advancements \
come new challenges and ethical dilemmas. As society grapples with the implications of AI, urgent questions \
about privacy, security, and the nature of consciousness itself come to the forefront. Amidst this dynamic \
backdrop, a new breakthrough in quantum computing promises to revolutionize the field even further. \
Scientists have developed a quantum processor capable of performing calculations at speeds previously thought \
impossible. This leap in technology opens the door to solving problems that have long stumped researchers, \
from predicting climate change patterns with unprecedented accuracy to unraveling the mysteries of the human \
genome. However, the power of this new technology also raises concerns about its potential misuse. Governments \
and corporations race to secure their own quantum capabilities, sparking a new kind of arms race. Meanwhile, \
a group of rogue programmers, known as the Shadow Collective, seeks to exploit the technology for their own \
ends. As tensions rise, a young scientist named Dr. Evelyn Zhang finds herself at the center of this \
unfolding drama. She has discovered a way to harness quantum computing to create a true artificial general \
intelligence (AGI), a machine capable of independent thought and reasoning. Dr. Zhang's creation, \
named Athena, possesses the potential to either save humanity from its own worst impulses or to become the \
ultimate instrument of control. As she navigates the treacherous waters of corporate espionage, government \
intrigue, and ethical quandaries, Dr. Zhang must decide the fate of her creation and, with it, the future \
of humanity. Will Athena be a benevolent guardian or a malevolent dictator? The answer lies in the choices \
made by those who wield its power. The world watches with bated breath as the next chapter in the saga of \
human and machine unfolds. In the midst of these global tensions, everyday life continues. Children \
attend schools where AI tutors provide personalized learning experiences. Hospitals use advanced \
algorithms to diagnose and treat patients with greater accuracy than ever before. The entertainment \
industry is transformed by virtual reality experiences that are indistinguishable from real life. Yet, \
for all the benefits, there are those who feel left behind by this technological revolution. Communities \
that once thrived on traditional industries find themselves struggling to adapt. The digital divide \
grows wider, creating new forms of inequality. Dr. Zhang's journey is not just a scientific quest but \
a deeply personal one. Her motivations are shaped by a desire to honor her late father's legacy, a \
pioneer in the field of AI who envisioned a future where technology would serve humanity's highest \
ideals. As she delves deeper into her research, she encounters allies and adversaries from unexpected \
quarters. A former colleague, Dr. Marcus Holt, now working for a rival tech giant, becomes both a \
rival and a potential ally as they navigate their complex relationship. In a hidden lab, far from \
prying eyes, Dr. Zhang and her team work tirelessly to refine Athena. They face numerous setbacks \
and breakthroughs, each step bringing them closer to their goal. The ethical implications of their \
work weigh heavily on them. Can a machine truly understand human emotions? Is it possible to program \
empathy and compassion? These questions haunt Dr. Zhang as she watches Athena's capabilities grow. \
As word of Athena's development leaks, the world reacts with a mixture of hope and fear. Protests \
erupt in major cities, with demonstrators demanding transparency and ethical oversight. Governments \
convene emergency sessions to discuss the potential impact of AGI on national security and global \
stability. Amid the chaos, the Shadow Collective launches a cyber-attack on Dr. Zhang's lab, attempting \
to steal her research. The attack is thwarted, but it serves as a stark reminder of the dangers they face. \
The final phase of Athena's development involves a series of tests to evaluate her decision-making \
abilities. Dr. Zhang designs scenarios that challenge Athena to balance competing interests and make \
ethical choices. In one test, Athena must decide whether to divert a runaway trolley to save a group \
of people at the expense of one individual. In another, she is tasked with allocating limited medical \
resources during a pandemic. Each test pushes the boundaries of machine ethics and highlights the \
complexities of programming morality. \n"

prompt_prefix="### Question:"

user_input=" Who is Evelyn Zhang?"

answer_prefix="\n### Answer:"

saved_prompt="${system_prompt}${rag_prefix}${rag}${prompt_prefix}"
echo -e "${saved_prompt}"

# save knowledge (it coule be done offline)
output1=$(curl -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d @- <<EOF
{
"model": "shenzhi-wang/Llama3.1-8B-Chinese-Chat",
"prompt": "${saved_prompt}",
"max_tokens": 2,
"temperature": 0
}
EOF
)

echo ""
echo "saved knowledge: $output1"
sleep 1

input_text="${saved_prompt}${user_input}${answer_prefix}"
echo -e "${input_text}"

output2=$(curl -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d @- <<EOF
{
"model": "shenzhi-wang/Llama3.1-8B-Chinese-Chat",
"prompt": "${input_text}",
"max_tokens": 64,
"temperature": 0
}
EOF
)

# Print the outputs of the curl requests
echo ""
echo "Output of request: $output2"

echo "Successfully finished test request!"
echo ""

# Cleanup commands, suppressing their output
ps -e | grep pt_main_thread | awk '{print $1}' | xargs kill -9 > /dev/null 2>&1
pkill -f python3 > /dev/null 2>&1
