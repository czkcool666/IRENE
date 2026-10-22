#!/bin/bash

usage() {
  echo "Usage: $0 [api|local]"
  exit 1
}

if [ "$#" -ne 1 ]; then
  usage
fi

mode=$1

if [ "$mode" = "api" ]; then
  echo "Running with API..."

  python ./gen_summary.py
  python ./translation.py \
    --model_name your_model \
    --method IRENE \
    --api_url your_api_url \
    --api_key your_api_key \
    --batch_size 1
  python ./Feedback.py \
    --model_name your_model \
    --api_url your_api_url \
    --api_key your_api_key \
    --batch_size 1

elif [ "$mode" = "local" ]; then
  echo "Running with local models..."

  models=(
      # single A100
      "deepseekcoder-1.3b"
      "deepseekcoder-6.7b"
      "Qwen2.5-Coder-1.5B-Instruct"
      "Qwen2.5-Coder-7B-Instruct"
      "codellama-7b"

      # double A100s
      "Qwen2.5-Coder-14B-Instruct"   
      "Qwen2.5-Coder-32B-Instruct" 
      "codellama-34b"
  )

  export CUDA_VISIBLE_DEVICES=2,3
  export VLLM_USE_V1=0

  for model in "${models[@]}"; do
      echo "Running model: $model"
      # python3 ./gen_summary.py --model_name ${model}
      python3 ./translation.py --model_name ${model} --method IRENE
      # python3 ./Feedback.py --model_name ${model} --method IRENE
  done

else
  echo "Invalid mode: $mode"
  usage
fi
