import os
import torch
import jsonlines
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
import argparse
import json
from static_analyzer import StaticAnalyzer
from datetime import datetime

class Infer:
    
    def __init__(self, model_name, api_url=None, api_key=None, batch_size=100, reasoning=False):
        self.model_name = model_name
        if api_url and api_key:
            self.api_url = api_url
            self.api_key = api_key
            self.use_openai = True
        else:
            self.use_openai = False
            self.tensor_parallel_size = torch.cuda.device_count()
            self.load_model()
        self.reasoning = reasoning
        self.batch_size = batch_size
        # ==============summary================ 
        self.USER_CONTEXT = (
            "Your task is to generate a **summary** based on **C code**.\n\n"
            "**Summarize the input, output, and overall function**\n"
            "### C Code\n"
            "```\n"
            "{c_code}\n"
            "```\n\n"
            "### Reminder\n"
            "**- Do not output any extra description or tokens other than the summarized text**\n"
        )
        
    def load_model(self):
        if hasattr(self, "model") and hasattr(self, "tokenizer"):
            return
        model_path = "your_model_path" + self.model_name
        if (
            "deepseekcoder" in self.model_name
            or "codellama" in self.model_name
            or "Qwen" in self.model_name
        ):
            self.model = LLM(
                model=model_path,
                trust_remote_code=True,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=15360, 
                swap_space=20,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            raise ValueError("Model not supported")
        
    def load_data(self):
        datasets = []
        dataset_path = (
                f'./dataset/groundtruth'
            )
        files = sorted(os.listdir(dataset_path))
        for file in files:
            full_path = os.path.join(dataset_path, file)
            if os.path.isdir(full_path):
                continue
                # print(f"Processing: {full_path}")
            with open(full_path, encoding='utf-8') as f:
                data = json.load(f)
                datasets.append(data)
        self.data = datasets

    # def load_data(self):
    #     datasets = []
    #     dataset_path = (
    #             f'your_data_dir'
    #         )
    #     filepath = os.path.join(dataset_path, "your_data.jsonl")
    #     data = list(jsonlines.open(filepath, mode='r'))
    #     for item in data:
    #         datasets.append(item)
    #     self.data = datasets    
    
    def construct_prompt(self, item):
        code = item["source_code"]
        prompt = self.USER_CONTEXT.format(
            # rules = rules,
            c_code=code,
        )
        return prompt
    
    def construct_prompts(self):
        if not hasattr(self, "data"):
            self.load_data()
        if self.use_openai:
            return [self.construct_prompt(item)
                    for item in self.data]
        prompts = []
        for item in self.data:
            inst = self.construct_prompt(item)
            prompt = self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": inst,
                    },
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
            # print(prompt)
            # input("Press Enter to continue...")
            prompts.append(prompt)
        return prompts
            
    def save_result(self, results):
        output_path = f"./middle_infer_result/summary/{self.model_name}"
        os.makedirs(output_path, exist_ok=True)
        with jsonlines.open(os.path.join(output_path, "infer_result.jsonl"), mode='w') as writer:
            for result in results:
                writer.write(result)
        print("Inference completed. Results saved to {}".format(output_path))
    
    def infer_vllm(self, prompts):
        results = []
        total_batches = (len(prompts) + self.batch_size - 1) // self.batch_size

        for idx in tqdm(range(0, len(prompts), self.batch_size), desc="Running vLLM inference", total=total_batches):
            batch_prompts = prompts[idx:idx + self.batch_size]
            sampling_params = SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=2048,
            )
            response = self.model.generate(batch_prompts, sampling_params)
            for i, res in enumerate(response):
                answer_content = []
                for j in range(1):
                    answer_content.append(res.outputs[j].text.strip())
                result={
                    "oai_response":{
                        "idx": idx + i,
                        "model": self.model_name,
                        "prompt": batch_prompts[i],
                        "response": answer_content
                    },
                    "source_data": self.data[idx + i]
                }
                results.append(result)

        self.save_result(results)
        
    def infer_openai(self, prompts):
        results = []
        client = OpenAI(api_key=self.api_key, base_url=self.api_url)

        def call_openai(idx, prompt):
            try:
                if self.reasoning:
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            # {"role": "system", "content": self.SYSTEM_CONTEXT},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        top_p=1,
                    )
                    reasoning_content = response.choices[0].message.reasoning_content.strip()
                    content = response.choices[0].message.content.strip()
                    answer_content = []
                    answer_content.append(content)
                    result={
                        "oai_response":{
                            "idx": idx,
                            "model": self.model_name,
                            "prompt": prompt,
                            "response": answer_content,
                            "reasoning": reasoning_content
                        },
                        "source_data": self.data[idx]
                    }
                else:
                    # print(prompt)
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            # {"role": "system", "content": self.SYSTEM_CONTEXT},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        max_tokens=2048,
                        top_p=1,
                    )
                    content = response.choices[0].message.content.strip()
                    # print(content)
                    # input("Press Enter to continue...")
                    answer_content = []
                    answer_content.append(content)
                    result={
                        "oai_response":{
                            "idx": idx,
                            "model": self.model_name,
                            "prompt": prompt,
                            "response": answer_content
                        },
                        "source_data": self.data[idx]
                    }
                return result
            except Exception as e:
                print(f"Error on prompt {idx}: {e}")
                return {
                    'idx': idx,
                    'prompt': prompt,
                    'response': f"Error: {e}"
                }

        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [executor.submit(call_openai, idx, prompt) for idx, prompt in enumerate(prompts)]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Running inference"):
                results.append(future.result())

        results.sort(key=lambda x: x['oai_response']['idx'])
        self.save_result(results)
        
    def infer(self):
        prompts = self.construct_prompts()
        if self.use_openai:
            self.infer_openai(prompts)
        else:
            self.infer_vllm(prompts)
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference with LLM')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--api_url', type=str, default=None, help='API URL for OpenAI')
    parser.add_argument('--api_key', type=str, default=None, help='API key for OpenAI')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for inference')
    parser.add_argument('--reasoning', action='store_true', help='Enable reasoning')
    args = parser.parse_args()
    infer_model = Infer(model_name=args.model_name, api_url=args.api_url, api_key=args.api_key, batch_size=args.batch_size, reasoning=args.reasoning)
    infer_model.infer()
