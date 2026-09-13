unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset no_proxy NO_PROXY
export HF_ENDPOINT=https://hf-mirror.com

hf download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3 --local-dir ~/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3