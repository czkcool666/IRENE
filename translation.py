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
import sys
import random
sys.path.append(os.path.dirname(__file__))

from Retrieve.bm25.bm25 import setup_bm25_retriever


class Infer:
    
    def __init__(self, model_name, method='IRENE', api_url=None, api_key=None, batch_size=1, reasoning=False):
        self.model_name = model_name
        self.method = method
        if self.method not in ['IRENE', 'Instruction', 'ICL', 'RAG', 'COT', 'Vert']:
            raise ValueError("Method must be one of: IRENE, Instruction, ICL, RAG, COT, Vert")
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
        if self.method == 'COT':
            self.SYSTEM_CONTEXT = (
                "You are an expert in both C and Rust. Your task is to translate C code into Rust. "
                "Please follow the output format of the examples and reply with the improved code in code blocks (```rust```) and explain the reasons briefly at the beginning."
            )
        else:
            self.SYSTEM_CONTEXT = (
                "You are an expert in both C and Rust. Your task is to translate C code into Rust. "
            )
        if self.method == 'IRENE':
            self.USER_CONTEXT_FEW_SHOT = (
                "I will provide C code, some examples, rules and summary.\n"
                "Your job is to translate **only** the C code into Rust.\n\n"
                "### Translation examples\n"
                "{few_shot}"
                "### C Code:\n"
                "```\n"
                "{c_code}\n"
                "```\n\n"
                "### Rules (Reference Only)\n"
                "This section may contains rules extracted from C Code above. Use it as **guidance** to ensure accurate translation of memory management, and type handling.\n"
                "{rules}"
                "### Summary (Reference Only)\n"
                "This section contains a summary about the C code. Use it as **guidance**.\n"
                "{summary}\n\n"
                "### Reminder\n"
                "1. Carefully read and understand both the C code, some examples, rules and summary.\n"
                "2. Output **only** the Rust code—explanations, comments, or additional text.\n" 
            )
            self.USER_CONTEXT_NO_FEW_SHOT = (
                "I will provide C code, rules and summary.\n"
                "Your job is to translate **only** the C code into Rust.\n\n"
                "### C Code:\n"
                "```\n"
                "{c_code}\n"
                "```\n\n"
                "### Rules (Reference Only)\n"
                "This section may contains rules extracted from C Code above. Use it as **guidance** to ensure accurate translation of memory management, and type handling.\n"
                "{rules}"
                "### Summary (Reference Only)\n"
                "This section contains a summary about the C code. Use it as **guidance**.\n"
                "{summary}\n\n"
                "### Reminder\n"
                "1. Carefully read and understand both the C code, rules and summary.\n"
                "2. Output **only** the Rust code—explanations, comments, or additional text.\n"
            )
        elif self.method == 'Instruction':
            self.USER_CONTEXT = (
                "Here is code in C programming lanaguge. Translate the following code from C to Rust programming lanaguge. Do not output any extra description or tokens other than the translated code.\n\n{c_code}"
            )
        elif self.method == 'ICL' or self.method == 'RAG' or self.method == 'COT':
            self.USER_CONTEXT_FEW_SHOT = (
                "I will provide C code and some examples.\n"
                "Your job is to translate **only** the C code into Rust.\n\n"
                "### Translation example\n"
                "{few_shot}\n"
                "### C Code:\n"
                "```\n"
                "{c_code}\n"
                "```\n\n"
                "### Reminder\n"
                "1. Carefully read and understand both the C code and example.\n"
                "2. Output **only** the translated Rust code—no explanations, comments, or additional text.\n"
            )
            self.USER_CONTEXT_NO_FEW_SHOT = (
                "I will provide C code.\n"
                "Your job is to translate **only** the C code into Rust.\n\n"
                "### C Code:\n"
                "```\n"
                "{c_code}\n"
                "```\n\n"
                "### Reminder\n"
                "1. Carefully read and understand the C code.\n"
                "2. Output **only** the translated Rust code—no explanations, comments, or additional text.\n"
            )
        elif self.method == 'Vert':
            self.VERT_CONTEXT = (
                "{c_code}\n"
                "\nCompilable Rust refactoring of above C code, with code only, no comments. Use the same function name, same argument types and return types. Make sure it includes all imports, uses safe rust, and compiles on its own. Give only code, and no main function. Convert i32 types to f32 if necessary. Use mut variables if necessary."
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
    
    def load_data(self):
        datasets = []
        dataset_path = (
                f'/data/luofeng/c2rust/code/middle_infer_result/summary/{self.model_name}'
            )
        filepath = os.path.join(dataset_path, "infer_result.jsonl")
        data = list(jsonlines.open(filepath, mode='r'))
        for item in data:
            if "oai_response" in item:
                item["source_data"]["agent1_output"] = item["oai_response"]["response"]
                item = item["source_data"]
            datasets.append(item)
        self.data = datasets
    
    def load_corpus(self):
        datasets = []
        filepath = 'your_corpus_path'  # Replace with your corpus path
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Corpus file not found at {filepath}")
        data = list(jsonlines.open(filepath, mode='r'))
        for item in data:
            datasets.append(item)
        self.corpus = datasets
    
    def count_tokens(self, text: str) -> int:
        tokenizer = self.tokenizer
        tokens = tokenizer.tokenize(text)
        return len(tokens)
    
    def build_few_shot_prompt(self, retrieve, k):
        few_shot = ""
        for i in range(min(k, len(retrieve))):
            example = retrieve[i]
            if example['score'] < 100:
                return few_shot
            few_shot += f"C code:\n```\n{example['c_code']}\n```\n"
            few_shot += f"This is the translated Rust code:\n```\n{example['rust_code']}\n```\n\n"
        return few_shot

    def construct_prompt(self, item):
        code = item["source_code"]
        #============few_shot construct===================
        #===========RAG===================
        if self.method == 'RAG':
            bm25_retriever = setup_bm25_retriever()
            bm25_retrieve = bm25_retriever(code)
            few_shot = self.build_few_shot_prompt(bm25_retrieve, k=1)
        #===========ICL, COT===================
        elif self.method == 'ICL' or self.method == 'COT':
            few_shot = ""
            # randomly select 4 examples from the corpus
            examples = random.sample(self.corpus, 4)
            for example in examples:
                example['score'] = 9999  # it ensures that this example will not be filtered out
            few_shot = self.build_few_shot_prompt(examples, k=4)
        #===========IRENE===================
        elif self.method == 'IRENE':
            analyzer = StaticAnalyzer('build/libtree-sitter-c.so', 'c', code)
            is_malloc = 0
            is_pointers = 0
            is_array = 0
            is_mixtype = 0
            is_io = 0
            
            # malloc
            if analyzer.extract_malloc_expressions()["malloc_vars"] != []:
                is_malloc = 1
                malloc_context = analyzer.gen_malloc_context()
            # pointers
            malloc_vars = analyzer.extract_malloc_expressions()["malloc_vars"]
            if analyzer.gen_pointer_context(malloc_vars):
                is_pointers = 1
                pointers_context = analyzer.gen_pointer_context(malloc_vars)
            # array
            if analyzer.gen_array_context():
                is_array = 1
                array_context = analyzer.gen_array_context()
            # mixed type operations
            symbol_table = analyzer.generate_symbol_table()
            if analyzer.detect_mixed_type_operations(symbol_table=symbol_table) != []:
                is_mixtype = 1
                mixtype_context = analyzer.detect_mixed_type_operations(symbol_table=symbol_table)
            # io
            io_context = analyzer.detect_multiple_scanf_inputs()
            if io_context != []:
                is_io = 1
                    
            with open('./Rules.jsonl', 'r', encoding='utf-8') as f:
                Demonstration = json.load(f)
            few_shot_prompts = {}
            rules = ""
            if is_malloc == 1:
                for malloc_json in malloc_context:
                    rules += f"{malloc_json}\n\n"
                # Retrieve example according to the rules
                category = 'malloc'
                bm25_retriever = setup_bm25_retriever(sample_path=f'/data/luofeng/c2rust/dataset/example/{category}.jsonl', cashe_path=f'bm25_cache_{category}.json')
                bm25_retrieve = bm25_retriever(code)
                few_shot_prompts[category] = self.build_few_shot_prompt(bm25_retrieve, k=1)
            if is_pointers == 1:
                for pointer_json in pointers_context:
                    rules += f"{pointer_json}\n\n"
                # Retrieve example according to the rules
                category = 'pointers'
                bm25_retriever = setup_bm25_retriever(sample_path=f'/data/luofeng/c2rust/dataset/example/{category}.jsonl', cashe_path=f'bm25_cache_{category}.json')
                bm25_retrieve = bm25_retriever(code)
                few_shot_prompts[category] = self.build_few_shot_prompt(bm25_retrieve, k=1)
            if is_array == 1:
                for array_json in array_context:
                    rules += f"{array_json}\n"
                rules += f"Tips: {Demonstration['arrays']['rules']}\n\n"
                # Retrieve example according to the rules
                category = 'array'
                bm25_retriever = setup_bm25_retriever(sample_path=f'/data/luofeng/c2rust/dataset/example/{category}.jsonl', cashe_path=f'bm25_cache_{category}.json')
                bm25_retrieve = bm25_retriever(code)
                few_shot_prompts[category] = self.build_few_shot_prompt(bm25_retrieve, k=1)
            if is_mixtype == 1:
                for mixtype_json in mixtype_context:
                    rules += f"{mixtype_json}\n"
                rules += f"Tips: {Demonstration['mixtype']['rules']}\n\n"
                # Retrieve example according to the rules
                category = 'mixtype'
                bm25_retriever = setup_bm25_retriever(sample_path=f'/data/luofeng/c2rust/dataset/example/{category}.jsonl', cashe_path=f'bm25_cache_{category}.json')
                bm25_retrieve = bm25_retriever(code)
                few_shot_prompts[category] = self.build_few_shot_prompt(bm25_retrieve, k=1)
            if is_io == 1:
                for io_json in io_context:
                    rules += f"{io_json}\n"
                rules += f"{Demonstration['io']['rules'][0]}\n\n"
                # Retrieve example according to the rules
                category = 'io'
                bm25_retriever = setup_bm25_retriever(sample_path=f'/data/luofeng/c2rust/dataset/example/{category}.jsonl', cashe_path=f'bm25_cache_{category}.json')
                bm25_retrieve = bm25_retriever(code)
                few_shot_prompts[category] = self.build_few_shot_prompt(bm25_retrieve, k=1)

            if few_shot_prompts == {}: # If there is no corresponding rule in the description, then search in the global demonstration library
                bm25_retriever = setup_bm25_retriever()
                bm25_retrieve = bm25_retriever(code)
                few_shot = self.build_few_shot_prompt(bm25_retrieve, k=1)
            else:
                few_shot = "".join(few_shot_prompts.values())


        ## ===========IRENE, RAG, ICL, COT===================
        if self.method == 'IRENE':
            if few_shot != "":
                prompt = self.USER_CONTEXT_FEW_SHOT.format(
                    few_shot = few_shot,
                    c_code=code,
                    rules = rules,
                    summary=item["agent1_output"][0],
                )
            else:
                prompt = self.USER_CONTEXT_NO_FEW_SHOT.format(
                    c_code=code,
                    rules = rules,
                    summary=item["agent1_output"][0],
                )
        elif self.method == 'RAG' or self.method == 'ICL' or self.method == 'COT':
            if few_shot != "":
                prompt = self.USER_CONTEXT_FEW_SHOT.format(
                    few_shot = few_shot,
                    c_code=code,
                )
            else:
                prompt = self.USER_CONTEXT_NO_FEW_SHOT.format(
                    c_code=code,
                )
        ## ===========Instruction and Vert===================
        elif self.method == 'Instruction':
            prompt = self.USER_CONTEXT.format(
                c_code=code,
            )
        elif self.method == 'Vert':
            prompt = self.VERT_CONTEXT.format(
                c_code=code,
            )

        return prompt
    
    def construct_prompts(self):
        if not hasattr(self, "data"):
            self.load_data()
            self.load_corpus()
        if self.use_openai:
            return [self.construct_prompt(item)
                    for item in self.data]
        prompts = []
        for item in self.data:
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
        output_path = f"./infer_result/{self.method}/{self.model_name}"
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
    parser.add_argument('--method', type=str, default='IRENE', help='The method to use for inference, e.g., IRENE, ICL, Instruction')
    parser.add_argument('--api_url', type=str, default=None, help='API URL for OpenAI')
    parser.add_argument('--api_key', type=str, default=None, help='API key for OpenAI')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for inference')
    parser.add_argument('--reasoning', action='store_true', help='Enable reasoning')
    args = parser.parse_args()
    infer_model = Infer(model_name=args.model_name, method = args.method, api_url=args.api_url, api_key=args.api_key, batch_size=args.batch_size, reasoning=args.reasoning)
    infer_model.infer()
