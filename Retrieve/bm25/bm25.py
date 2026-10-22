import json
from rank_bm25 import BM25Okapi
from typing import List, Tuple
import os
import hashlib
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(file_path: str) -> List[dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def preprocess(text: str) -> List[str]:
    return text.lower().split()


def build_bm25_index(docs: List[str]) -> BM25Okapi:
    tokenized_corpus = [preprocess(doc) for doc in docs]
    return BM25Okapi(tokenized_corpus)


def bm25_retrieve(bm25: BM25Okapi, docs: List[dict], query: str, k: int = 10) -> List[Tuple[dict, float]]:
    tokenized_query = preprocess(query)
    scores = bm25.get_scores(tokenized_query)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(docs[i], scores[i]) for i in top_k_indices]


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def save_cache(cache_path: str, cache_data: dict):
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)


def load_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def bm25_result(query, data, cache_path, top_k=10,):
    query_hash = hash_text(query)
    cache = load_cache(cache_path)
    if query_hash in cache:
        return cache[query_hash]  # 已缓存

    texts = [item["c_code"] for item in data]
    bm25 = build_bm25_index(texts)
    results = bm25_retrieve(bm25, data, query, k=top_k)
    # 保存 top_k 条结果（文本和得分）
    formatted = [{"c_code": item[0]["c_code"], "rust_code": item[0]["rust_code"], "score": item[1]} for item in results]
    cache[query_hash] = formatted
    save_cache(cache_path, cache)
    return formatted


def load_data():
    datasets = []
    dataset_path = "/data/luofeng/c2rust/dataset/groundtruth"
    files = sorted(os.listdir(dataset_path))
    for file in files:
        full_path = os.path.join(dataset_path, file)
        if os.path.isdir(full_path):
            continue
        with open(full_path) as f:
            data = json.load(f)
            datasets.append(data)
    return datasets

def build_few_shot_prompt(bm25_retrieve, k):
    few_shot = ""
    for i in range(min(k, len(bm25_retrieve))):
        example = bm25_retrieve[i]
        few_shot += f"C code:\n```\n{example['c_code']}\n```\n"
        few_shot += f"This is the translated Rust code:\n```\n{example['rust_code']}\n```\n\n"
    return few_shot

def setup_bm25_retriever(base_dir=None, sample_path='/data/luofeng/c2rust/dataset/example/successful.jsonl', cashe_path = 'bm25_cache_tmp2.json', top_k=10):
    """
    初始化并返回一个可用的 retriever 函数
    """
    if base_dir is None:
        base_dir = os.path.dirname(__file__)

    # cache_path = os.path.join(base_dir, "bm25_cache.json") # 第一版fewshot结果复现缓存
    # cache_path = os.path.join(base_dir, "bm25_cache_tmp.json") # 第二版fewshot结果复现缓存
    cache_path = os.path.join(base_dir, cashe_path) # 第三版fewshot结果复现缓存
    data = load_jsonl(sample_path)
    def retriever(query_code: str):
        return bm25_result(query_code, data, cache_path, top_k)

    return retriever

def plot_score_distribution(query: str, jsonl_path: str, bins: int = 20, save_path: str = "score_distribution.png"):
    data = load_jsonl(jsonl_path)
    texts = [item["c_code"] for item in data]
    bm25 = build_bm25_index(texts)
    scores = bm25.get_scores(preprocess(query))

    

    # 归一化 scores 到 [0, 1]
    min_score = min(scores)
    max_score = max(scores)
    if max_score > min_score:
        normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
    else:
        normalized_scores = [0.0 for _ in scores]  # 如果全是同一个分数

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.hist(
        normalized_scores,
        bins=np.linspace(0, 1, 41),  # 细化到0.025一个bin
        edgecolor='black',
        color='skyblue'
    )
    plt.xticks(np.linspace(0, 1, 11))  # 显示0.0 ~ 1.0 每0.1刻度
    plt.xlabel("BM25 Score")
    plt.ylabel("Number of Samples")
    plt.title("BM25 Score Distribution for Query")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_top1_score_distribution(data: list, jsonl_path: str, bins: int = 20, save_path: str = "top1_score_distribution.pdf", normalize: bool = True):
    samples = load_jsonl(jsonl_path)
    texts = [item["c_code"] for item in samples]
    cache_path = "bm25_cache_tmp2.json"
    top1_scores = []
    for item in data:
        results = bm25_result(item["source_code"], texts, cache_path)
        top1_score = results[0]["score"]
        top1_scores.append(top1_score)

    thresholds = [0, 50, 100, 150, 200, 250, 300, 350, 400]
    total = len(top1_scores)

    print("大于 0 / 50 / 100 / 150 ...的占比如下：")
    for t in thresholds:
        count = sum(score > t for score in top1_scores)
        ratio = count / total if total > 0 else 0
        print(f"分数 > {t}: {ratio:.2%} ({count}/{total})")
    
    # 归一化（可选）
    if normalize and top1_scores:
        min_score = min(top1_scores)
        max_score = max(top1_scores)
        if max_score > min_score:
            top1_scores = [(s - min_score) / (max_score - min_score) for s in top1_scores]
        else:
            top1_scores = [0.0 for _ in top1_scores]

    # 画图
    plt.figure(figsize=(10, 6))
    plt.hist(top1_scores, bins=bins, edgecolor='black', color='lightcoral')
    plt.xlabel("Normalized Top-1 BM25 Score" if normalize else "Top-1 BM25 Score")
    plt.ylabel("Number of Samples")
    plt.title("Top-1 BM25 Score Distribution Across Dataset")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# === 使用示例 ===
if __name__ == "__main__":
    jsonl_path = "/data/luofeng/c2rust/dataset/example/successful.jsonl"
    # cache_path = "bm25_cache.json"
    # cache_path = "bm25_cache_tmp.json"
    cache_path = "bm25_cache_tmp2.json"
    data = load_data()
    plot_top1_score_distribution(data, jsonl_path, bins=30, save_path="top1_score_distribution_tmp_1_normalize.pdf")
    # for item in data:
    #     results = bm25_result(item["source_code"], jsonl_path, top_k=10, cache_path=cache_path)
    #     few_shot = build_few_shot_prompt(results, k=2)
    #     print(f"Few-shot prompt:\n{few_shot}")
    #     input("Press Enter to continue...")

    # 示例：绘制某个 query 的分数分布
    # example_query = data[1]["source_code"]
    # plot_score_distribution(example_query, jsonl_path, bins=50, save_path="bm25.pdf")
