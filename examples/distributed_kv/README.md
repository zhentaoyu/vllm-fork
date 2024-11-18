
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