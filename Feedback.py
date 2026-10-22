import os
import torch
import jsonlines
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
import argparse
import sys
from compile_check import process_item
import re
sys.path.append(os.path.dirname(__file__))



class Infer:
    
    def __init__(self, model_name, method='IRENE', api_url=None, api_key=None, batch_size=1, reasoning=False):
        self.model_name = model_name
        self.method = method
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
        self.SYSTEM_CONTEXT = (
            "You are an expert in Rust. Your task is to generate correct and safe Rust."
        )
        self.USER_CONTEXT = (
            "Here is a Rust code you generated\n{rust_code}\n."
            "The error is \n{error}\n. "
            "Generate safe rust again. Do not output any extra description or tokens other than the safe Rust."
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
                max_model_len=15384, 
                swap_space=20,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            raise ValueError("Model not supported")
        
    def post_process(self, infer_result):
        match = re.search(r"```(?:\w+)?\n(.*?)```", infer_result, re.DOTALL)
        code = match.group(1).strip() if match else infer_result.strip()
        return code
    
    def load_data(self):
        error_datasets = []
        success_datasets = []
        dataset_path = (
                f'./infer_result/IRENE/{self.model_name}'
            )
        filepath = os.path.join(dataset_path, "infer_result.jsonl")
        data = list(jsonlines.open(filepath, mode='r'))
        for item in data:
            rust_code = item["oai_response"]["response"][0]
            compile_code, compile_text = process_item(rust_code)
            if compile_code == 0:     # 考虑warning信息
                success_datasets.append(item)
            else:
                item["source_data"]["rust_code"] = self.post_process(rust_code)
                item["source_data"]["compile_code"] = compile_code
                item["source_data"]["compile_text"] = compile_text
                item["source_data"]["idx"] = item["oai_response"]["idx"]
                item = item["source_data"]
                error_datasets.append(item)

        self.error_data = error_datasets
        self.success_data = success_datasets
    
    def count_tokens(self, text: str) -> int:
        tokenizer = self.tokenizer
        tokens = tokenizer.tokenize(text)
        return len(tokens)

    def construct_prompt(self, item):
        rust_code = item["rust_code"]
        error = item["compile_text"]

        prompt = self.USER_CONTEXT.format(
            rust_code=rust_code,
            error = error,
        ) 
        return prompt
    
    def construct_prompts(self):
        if not hasattr(self, "error_data"):
            self.load_data()
        if self.use_openai:
            return [self.construct_prompt(item)
                    for item in self.error_data]
        prompts = []
        for item in self.error_data:
            inst = self.construct_prompt(item)
            if "starcoder" in self.model_name:
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
            else:
                prompt = self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": self.SYSTEM_CONTEXT,
                        },
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
        output_path = f"./infer_result/IRENE/{self.model_name}"
        os.makedirs(output_path, exist_ok=True)
        with jsonlines.open(os.path.join(output_path, "infer_result_fixed.jsonl"), mode='w') as writer:
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
                        "idx": self.error_data[idx + i]["idx"],
                        "model": self.model_name,
                        "prompt": batch_prompts[i],
                        "response": answer_content
                    },
                    "source_data": self.error_data[idx + i]
                }
                results.append(result)
        results.extend(self.success_data)
        results.sort(key=lambda x: x['oai_response']['idx'])
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
                            {"role": "system", "content": self.SYSTEM_CONTEXT},
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
                            "idx": self.error_data[idx]["idx"],
                            "model": self.model_name,
                            "prompt": prompt,
                            "response": answer_content,
                            "reasoning": reasoning_content
                        },
                        "source_data": self.error_data[idx]
                    }
                else:
                    # print(prompt)
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_CONTEXT},
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
                            "idx": self.error_data[idx]['idx'],
                            "model": self.model_name,
                            "prompt": prompt,
                            "response": answer_content
                        },
                        "source_data": self.error_data[idx]
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

        # results.sort(key=lambda x: x['oai_response']['idx'])
        results.extend(self.success_data)
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
    parser.add_argument('--method', type=str, default='IRENE', help='The method to use for inference, e.g., IRENE, ICL, Instruction')
    parser.add_argument('--api_url', type=str, default=None, help='API URL for OpenAI')
    parser.add_argument('--api_key', type=str, default=None, help='API key for OpenAI')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for inference')
    parser.add_argument('--reasoning', action='store_true', help='Enable reasoning')
    args = parser.parse_args()
    infer_model = Infer(model_name=args.model_name, method = args.method, api_url=args.api_url, api_key=args.api_key, batch_size=args.batch_size, reasoning=args.reasoning)
    infer_model.infer()
