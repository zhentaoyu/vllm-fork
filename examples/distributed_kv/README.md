
## docker build
```shell
sudo docker build -f Dockerfile.hpukv -t vllm-hpukv .
```

## docker run
```shell
docker run -it --rm --runtime=habana --name="vllm-kv" -p 32769:8000 \
-e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host \
-e HTTPS_PROXY=$https_proxy -e HTTP_PROXY=$https_proxy \
-e HF_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
-e VLLM_SKIP_WARMUP=true -e LLM_KV_TRANSFER_DRIVER="disk_kv_transfer" \
vllm-hpukv:latest
```

## v1/completions

curl -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
"prompt": "San Francisco is a",
"max_tokens": 10,
"temperature": 0
}'

curl -s http://100.83.111.240:32769/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
"prompt": "San Francisco is a",
"max_tokens": 10,
"temperature": 0
}'

## v1/chat/completions

curl -s http://100.83.111.240:32769/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
"messages": [{"role": "user", "content": "What is deep learning?"}],
"max_completion_tokens": 10,
"temperature": 0
}'

curl 100.83.111.240:32769/v1/chat/completions  -X POST   -d '{"model":"meta-llama/Meta-Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "What is deep learning?"}]}'   -H 'Content-Type: application/json'

curl 10.106.53.93:9009/v1/chat/completions  -X POST   -d '{"model":"meta-llama/Meta-Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "\n### You are a helpful, respectful and honest assistant to help the user with questions.Please refer to the search results obtained from the local knowledge base. But be careful to not incorporate the information that you think is not relevant to the question. If you dont know the answer to a question, please dont share false information. \n\n### Search results:  \n\n### Question: what is artificial intelligence. \n\n### Answer:\n"}], "max_tokens":256, "n": 1, "stream":false}'   -H 'Content-Type: application/json'

## OPEA

curl 10.107.4.218:8888/v1/chatqna -H "Content-Type: application/json" -d '{"messages": "what is artificial intelligence.", "max_tokens":128}'