# Qwen3 8B 

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-8B \
    --port 8900 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.4 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 32768 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes

prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 -m Qwen/Qwen3-8B -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m Qwen/Qwen3-8B -b http://0.0.0.0:8900/v1

# Qwen3 8B (willcb)

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve willcb/Qwen3-8B \
    --port 8900 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.5 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 32768 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes

prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 -m willcb/Qwen3-8B -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m willcb/Qwen3-8B -b http://0.0.0.0:8900/v1

# Qwen3 14B

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-14B \
    --port 8900 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.7 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 32768 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes

prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 -m Qwen/Qwen3-14B -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m Qwen/Qwen3-14B -b http://0.0.0.0:8900/v1


# Qwen3 32B

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-32B \
    --port 8900 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 32768 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes

prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 -m Qwen/Qwen3-32B -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m Qwen/Qwen3-32B -b http://0.0.0.0:8900/v1

# Qwen3 32B (willcb)

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve willcb/Qwen3-32B \
    --port 8900 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 32768 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes

prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 -m willcb/Qwen3-32B -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m willcb/Qwen3-32B -b http://0.0.0.0:8900/v1

# Qwen3-Coder-Next 80B (3B active)

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-Coder-Next \
    --port 8900 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 32768 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder


# Devstral 2 Small

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve mistralai/Devstral-Small-2-24B-Instruct-2512 \
    --port 8900 \
    --data-parallel-size 4 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.75 \
    --enforce-eager \
    --max-model-len 65536 \
    --tool-call-parser mistral --enable-auto-tool-choice

prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 -m mistralai/Devstral-Small-2-24B-Instruct-2512 -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m mistralai/Devstral-Small-2-24B-Instruct-2512 -b http://0.0.0.0:8900/v1

# GLM 4.7 Flash


CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve zai-org/GLM-4.7-Flash \
    --port 8900 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.80 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 65536 \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 1 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --served-model-name glm-4.7-flash

prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 -m zai-org/GLM-4.7-Flash -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m zai-org/GLM-4.7-Flash -b http://0.0.0.0:8900/v1

# Nanbeige/Nanbeige4.1-3B

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Nanbeige/Nanbeige4.1-3B \
    --port 8900 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.7 \
    --dtype bfloat16 \
    --enforce-eager \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-model-len 65536

prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -r 3 -m Nanbeige/Nanbeige4.1-3B -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m Nanbeige/Nanbeige4.1-3B -b http://0.0.0.0:8900/v1

uv run rl @ configs/prime-rl/arc-agi-nanbeige.toml

# Kill GPU processes
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9

# OpenRouter 

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 \
    -m arcee-ai/trinity-large-preview:free \
    -b https://openrouter.ai/api/v1 \
    -k OPENROUTER_API_KEY

# gpt-oss-120b

docker run --rm --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  -p 8900:8900 \
  --ipc=host \
  vllm/vllm-openai:latest \
  openai/gpt-oss-120b \
  --port 8900 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 65536 \
  --tool-call-parser openai

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve openai/gpt-oss-120b \
    --port 8900 \
    --async-scheduling \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.80 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --enable-auto-tool-choice --tool-call-parser openai \
    --enforce-eager

prime eval run arc-agi -x '{"data_dir":"data/arc-dummy"}' -n 1 -m openai/gpt-oss-120b -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"data_dir":"data/arc-prize-2024"}' -n 4 -r 3 -m openai/gpt-oss-120b -b http://0.0.0.0:8900/v1

# Nemotron Cascade 14B

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve nvidia/Nemotron-Cascade-14B-Thinking \
    --port 8900 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.5 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 32768 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes

prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 -m nvidia/Nemotron-Cascade-14B-Thinking -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m nvidia/Nemotron-Cascade-14B-Thinking -b http://0.0.0.0:8900/v1

# Liquid AI

prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 \
    -m liquid/lfm-2.5-1.2b-thinking:free \
    -b https://openrouter.ai/api/v1 \
    -k OPENROUTER_API_KEY


prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 \
    -m liquid/lfm-2.5-1.2b-thinking:free \
    -b https://openrouter.ai/api/v1 \
    -k OPENROUTER_API_KEY

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 \
    -m qwen/qwen3.5-flash-02-23 \
    -b https://openrouter.ai/api/v1 \
    -k OPENROUTER_API_KEY

# Teacher


prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m Qwen/Qwen3.5-27B -b http://0.0.0.0:8907/v1


# Qwen3.5-27B

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8900:8900 \
    --ipc=host \
    vllm/vllm-openai:qwen3_5 \
    --port 8900 \
    --model Qwen/Qwen3.5-27B \
    --async-scheduling \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.80 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --enable-prefix-caching \
    --language-model-only \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes


prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 -m Qwen/Qwen3.5-27B -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m Qwen/Qwen3.5-27B -b http://0.0.0.0:8900/v1

# Qwen3.5-9B

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8907:8907 \
    --ipc=host \
    vllm/vllm-openai:qwen3_5 \
    --port 8907 \
    --model Qwen/Qwen3.5-9B \
    --async-scheduling \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.80 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --enable-prefix-caching \
    --language-model-only \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes


prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 -m Qwen/Qwen3.5-9B -b http://0.0.0.0:8907/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m Qwen/Qwen3.5-9B -b http://0.0.0.0:8907/v1

# Fix flash-attn issue
uv sync --extra flash-attn

# Teacher

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve willcb/Qwen3-32B \
    --port 8932 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 32768 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve willcb/Qwen3-32B \
    --port 8932 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 32768 \
    --enable-auto-tool-choice --tool-call-parser hermes

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m willcb/Qwen3-32B -b http://0.0.0.0:8932/v1

# Qwen3.5-27B

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3.5-27B \
    --port 8907 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --language-model-only \
    --enforce-eager \
    --enable-prefix-caching

prime eval run arc-agi -a '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m Qwen/Qwen3.5-27B -b http://0.0.0.0:8907/v1

# Qwen3.5-9B

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8907:8907 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --port 8907 \
    --model Qwen/Qwen3.5-9B \
    --async-scheduling \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.80 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --reasoning-parser qwen3 \
    --enable-prefix-caching


CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3.5-9B \
    --port 8907 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager \
    --enable-prefix-caching 

prime eval run arc-agi -a '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m Qwen/Qwen3.5-9B -b http://0.0.0.0:8907/v1

# Qwen3.5-35B-A3B

docker run --runtime nvidia --gpus 1,2 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8935:8935 \
    --ipc=host \
    vllm/vllm-openai \
    --port 8935 \
    --model Qwen/Qwen3.5-35B-A3B \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder


docker run --runtime nvidia --gpus 1,2 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8935:8935 \
    --ipc=host \
    vllm/vllm-openai \
    --port 8935 \
    --model Qwen/Qwen3.5-35B-A3B \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --reasoning-parser qwen3 \
    --enable-prefix-caching


# Qwen3.5-27B


CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-27B \
    --port 8927 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --reasoning-parser qwen3 \
    --enforce-eager \
    --enable-prefix-caching

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3.5-27B \
    --port 8927 \
    --gpu-memory-utilization 0.85 \
    --data-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --reasoning-parser qwen3 \
    --enforce-eager \
    --enable-prefix-caching


docker run --runtime nvidia --gpus 0 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8927:8927 \
    --ipc=host \
    vllm/vllm-openai:qwen3_5 \
    --port 8927 \
    --model Qwen/Qwen3.5-27B \
    --async-scheduling \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 


docker run --runtime nvidia --gpus '"device=0,1"' \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8927:8927 \
    --ipc=host \
    vllm/vllm-openai:qwen3_5 \
    --port 8927 \
    --data-parallel-size 2 \
    --model Qwen/Qwen3.5-27B \
    --async-scheduling \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8927:8927 \
    --ipc=host \
    vllm/vllm-openai:qwen3_5 \
    --port 8927 \
    --data-parallel-size 3 \
    --model Qwen/Qwen3.5-27B \
    --async-scheduling \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 

# baidu/Qianfan-OCR

docker run --runtime nvidia --gpus '"device=1"' \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8904:8904 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --port 8904 \
    --model baidu/Qianfan-OCR \
    --async-scheduling \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --trust-remote-code \
    --hf-overrides '{"architectures":["InternVLChatModel"],"model_type":"internvl_chat"}'

# Qwen/Qwen3.5-122B-A10B 


CUDA_VISIBLE_DEVICES=0,1,2 vllm serve Qwen/Qwen3.5-122B-A10B \
    --port 8927 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 3 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --reasoning-parser qwen3 \
    --enforce-eager \
    --enable-prefix-caching


# Qwen/Qwen3.5-9B

CUDA_VISIBLE_DEVICES=0,1,2 vllm serve Qwen/Qwen3.5-9B \
    --port 8909 \
    --gpu-memory-utilization 0.8 \
    --data-parallel-size 3 \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --reasoning-parser qwen3 \
    --enforce-eager \
    --enable-prefix-caching

# Mistral Small 3.2 24B (Vision/Multimodal VLM for DocVQA)
# Apache 2.0 license, DocVQA 94.86%, ChartQA 87.40%
# ~48GB VRAM in bf16, fits on 1x A100 80GB
# Uses mistral tokenizer mode + config format for vision support
# Docs: https://docs.mistral.ai/deployment/self-deployment/vllm
# HF: https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506

CUDA_VISIBLE_DEVICES=0 vllm serve mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
    --port 8924 \
    --tokenizer_mode mistral \
    --config_format mistral \
    --load_format mistral \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"image":10}' \
    --enforce-eager

# Qwen2.5-VL-72B (Vision VLM for DocVQA)
# ~144GB VRAM in bf16, needs TP=2 on A100 80GB
# Docs: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html

CUDA_VISIBLE_DEVICES=0,1,2 vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
    --port 8972 \
    --tensor-parallel-size 3 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --mm-encoder-tp-mode data \
    --enforce-eager \
    --limit-mm-per-prompt '{"image":10}'


# Qwen 27B on other server 

ssh -N -L 8928:localhost:8928 144.122.52.26

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3.5-27B \
    --port 8928 \
    --gpu-memory-utilization 0.85 \
    --data-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --reasoning-parser qwen3 \
    --enforce-eager \
    --enable-prefix-caching

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8928:8928 \
    --ipc=host \
    vllm/vllm-openai:qwen3_5 \
    --port 8928 \
    --model Qwen/Qwen3.5-27B \
    --data-parallel-size 4 \
    --async-scheduling \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8928:8928 \
    --ipc=host \
    vllm/vllm-openai:qwen3_5 \
    --port 8928 \
    --model Qwen/Qwen3.5-27B \
    --data-parallel-size 2 \
    --tensor-parallel-size 2 \
    --async-scheduling \
    --gpu-memory-utilization 0.85 \
    --max-model-len 131072 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 


docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8927:8927 \
    --ipc=host \
    vllm/vllm-openai:qwen3_5 \
    --port 8927 \
    --model Qwen/Qwen3.5-27B \
    --tensor-parallel-size 2 \
    --async-scheduling \
    --gpu-memory-utilization 0.85 \
    --max-model-len 131072 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 

# Gemma 4 31B

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8931:8931 \
    --ipc=host \
    vllm/vllm-openai:gemma4 \
    --port 8931 \
    --model google/gemma-4-31B-it \
    --reasoning-parser gemma4 \
    --data-parallel-size 3 \
    --max-model-len 131072 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --shm-size 16G \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8931:8931 \
    --ipc=host \
    vllm/vllm-openai:gemma4-cu130 \
    --port 8931 \
    --model google/gemma-4-31B-it \
    --enforce-eager \
    --async-scheduling \
    --reasoning-parser gemma4 \
    --data-parallel-size 3 \
    --max-model-len 131072 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90


CUDA_VISIBLE_DEVICES=0,1,2 vllm serve google/gemma-4-31B-it \
    --port 8931 \
    --data-parallel-size 3 \
    --max-model-len 131072 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90

CUDA_VISIBLE_DEVICES=0 vllm serve google/gemma-4-31B-it \
    --port 8831 \
    --max-model-len 131072 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90

docker run --runtime nvidia --gpus 'device=0' \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8831:8831 \
    --ipc=host \
    vllm/vllm-openai:gemma4 \
    --port 8831 \
    --model google/gemma-4-31B-it \
    --max-model-len 131072 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 

# Qwen3.5-27B

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8927:8927 \
    --ipc=host \
    vllm/vllm-openai \
    --port 8927 \
    --model Qwen/Qwen3.5-27B \
    --data-parallel-size 4 \
    --async-scheduling \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --reasoning-parser qwen3 

docker run --runtime nvidia --gpus '"device=0,1"' \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env NVIDIA_VISIBLE_DEVICES=0 \
    -p 8927:8927 \
    --ipc=host \
    vllm/vllm-openai \
    --port 8927 \
    --model Qwen/Qwen3.5-27B \
    --async-scheduling \
    --data-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --reasoning-parser qwen3 

CUDA_VISIBLE_DEVICES=0,1,2 vllm serve Qwen/Qwen3.5-27B \
    --port 8927 \
    --gpu-memory-utilization 0.85 \
    --data-parallel-size 3 \
    --async-scheduling \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --reasoning-parser qwen3

# Qwen/Qwen3.6-35B-A3B

CUDA_VISIBLE_DEVICES=0,1,2 vllm serve Qwen/Qwen3.6-35B-A3B \
    --port 8935 \
    --gpu-memory-utilization 0.85 \
    --data-parallel-size 3 \
    --async-scheduling \
    --enable-prefix-caching \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --reasoning-parser qwen3

docker run --runtime nvidia --gpus '"device=2"' \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8935:8935 \
    --ipc=host \
    vllm/vllm-openai \
    --port 8935 \
    --model Qwen/Qwen3.6-35B-A3B \
    --async-scheduling \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --reasoning-parser qwen3 

# Qwen3.6-27B

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8928:8928 \
    --ipc=host \
    vllm/vllm-openai \
    --port 8928 \
    --model Qwen/Qwen3.6-27B \
    --data-parallel-size 3 \
    --async-scheduling \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --reasoning-parser qwen3 

docker run --runtime nvidia --gpus '"device=0,1"' \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env NVIDIA_VISIBLE_DEVICES=0 \
    -p 8928:8928 \
    --ipc=host \
    vllm/vllm-openai \
    --port 8928 \
    --model Qwen/Qwen3.6-27B \
    --async-scheduling \
    --data-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --reasoning-parser qwen3 

CUDA_VISIBLE_DEVICES=0,1,2 vllm serve Qwen/Qwen3.6-27B \
    --port 8928 \
    --gpu-memory-utilization 0.85 \
    --data-parallel-size 3 \
    --async-scheduling \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --reasoning-parser qwen3

docker run --runtime nvidia --gpus '"device=2"' \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8928:8928 \
    --ipc=host \
    vllm/vllm-openai \
    --port 8928 \
    --model Qwen/Qwen3.6-27B \
    --async-scheduling \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --reasoning-parser qwen3 

# Qwen3.5-27B with tool calling

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8927:8927 \
    --ipc=host \
    vllm/vllm-openai \
    --port 8927 \
    --model Qwen/Qwen3.5-27B \
    --data-parallel-size 4 \
    --async-scheduling \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --reasoning-parser qwen3 

# ============================================================
# VLM-OPTIMIZED serving — DocVQA rvlm (recursive batch_look) on 80GB A100
# Added 2026-07-02. rvlm re-queries the SAME page image many times per doc
# (crop/zoom loop), so the multimodal *processor cache* is the big lever —
# it caches image preprocessing (resize/patchify) across those repeated calls.
# ============================================================

# Small model (4B / 9B) AS THE VLM, one 80GB GPU per replica (DP, not TP).
# COPY-PASTE SAFE: no inline comments — a `\` must be the LAST char on its line
# or the shell breaks the continuation (that's what gave you
# `--limit-mm-per-prompt: command not found`). Flag explanations are BELOW.
# Example filled for 4B on GPU1, port 8904, DP=1:
docker run -d --runtime nvidia --gpus '"device=1"' --ipc=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8904:8904 --name qwen35-4b \
    vllm/vllm-openai:qwen3_5 \
    --port 8904 \
    --model Qwen/Qwen3.5-4B \
    --data-parallel-size 1 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 768 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --enable-prefix-caching \
    --mm-processor-cache-type shm \
    --limit-mm-per-prompt '{"image":8}' \
    --async-scheduling \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder

# Bare-CLI form (no docker): `vllm serve` wants the model POSITIONAL, not --model
# (that's the deprecation warning you saw). Same flags:
#   CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3.5-4B --port 8904 \
#     --data-parallel-size 1 --gpu-memory-utilization 0.92 --max-num-seqs 768 \
#     --dtype bfloat16 --max-model-len 65536 --enable-prefix-caching \
#     --mm-processor-cache-type shm --limit-mm-per-prompt '{"image":8}' \
#     --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder

# Flag notes (keep these OUT of the command):
# - --data-parallel-size N : 1 full replica/GPU; DP >> TP for a model that fits on 1 GPU.
# - --gpu-memory-utilization 0.92 : 4B weights ~8GB of 80GB -> rest becomes KV cache.
# - --max-num-seqs 768 : THE throughput lever for a small model (hundreds of concurrent seqs).
# - --max-model-len 65536 : plenty for rvlm-on-val; larger over-reserves blocks, caps concurrency.
# - --enable-prefix-caching : rvlm reuses one system prompt every call -> computed once.
# - --mm-processor-cache-type shm : *** VLM KEY *** caches processed images across the
#     repeated batch_look calls on the same page (and across DP replicas).
# - --limit-mm-per-prompt '{"image":8}' : rvlm sends 1 image/batch_look; small limit avoids
#     reserving mm cache for nothing. Use image:32+ ONLY for the multi-image baselines
#     (official_baseline / raw_vlm_multi).
# - TP VLMs only (big VLM, TP>1): also add `--mm-encoder-tp-mode data`. No-op for DP-on-1-GPU.
# - Client-side eval concurrency can go high (c=24+) once the VLM has this config.
