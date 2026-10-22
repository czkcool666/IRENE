import os
import json
import jsonlines
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import hashlib
import matplotlib.pyplot as plt

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


def compute_and_save_embeddings(model_path, jsonl_path, output_code_path, output_vec_path, faiss_index_path):
    model = SentenceTransformer(model_path)
    data = list(jsonlines.open(jsonl_path, mode='r'))
    code_list = [item["c_code"] for item in data]

    embeddings = model.encode(code_list, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    # 保存代码
    with jsonlines.open(output_code_path, mode='w') as writer:
        for code in code_list:
            writer.write({"code": code})

    # 保存嵌入向量
    np.save(output_vec_path, embeddings)

    # 构建 FAISS 索引并保存
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 使用内积 = 归一化后的余弦相似度
    index.add(embeddings)
    faiss.write_index(index, faiss_index_path)


def load_code_and_index(code_path, vec_path, index_path):
    # 加载代码
    code_list = []
    with jsonlines.open(code_path, mode='r') as reader:
        for obj in reader:
            code_list.append(obj["code"])

    # 加载向量和索引
    embeddings = np.load(vec_path)
    index = faiss.read_index(index_path)
    return code_list, embeddings, index


def find_top_k_faiss(query_code, model, index, code_list, k=10):
    query_vec = model.encode([query_code], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(query_vec, k)
    result = []
    for score, idx in zip(scores[0], indices[0]):
        result.append((code_list[idx], float(score)))
    return result

def normalize_code(code: str) -> str:
    return ' '.join(code.strip().split())


def hash_text(text: str) -> str:
    """对标准化代码生成哈希值（作为 key）"""
    return hashlib.md5(normalize_code(text).encode('utf-8')).hexdigest()


def save_cache(cache_path: str, cache_data: dict):
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)


def load_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def query_with_cache(query_code: str,
                     model: SentenceTransformer,
                     faiss_index,
                     code_data: list,
                     cache_path: str,
                     top_k=10):
    query_hash = hash_text(query_code)
    cache = load_cache(cache_path)

    if query_hash in cache:
        return cache[query_hash]

    # FAISS 检索
    query_vec = model.encode([query_code], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = faiss_index.search(query_vec, top_k)
    results = []

    for score, idx in zip(scores[0], indices[0]):
        entry = {
            "c_code": code_data[idx]["c_code"],
            "rust_code": code_data[idx]["rust_code"],
            "score": float(score)
        }
        results.append(entry)

    # 保存缓存
    cache[query_hash] = results
    save_cache(cache_path, cache)

    return results

def query_with_cache_no_model(query_code: str, cache_path: str):
    query_hash = hash_text(query_code)
    cache = load_cache(cache_path)
    if query_hash in cache:
        return cache[query_hash]
    else:
        # 如果缓存中没有，则返回空列表
        print(f"Query code not found in cache: {query_code}")
        return []
   

def setup_retriever(base_dir=None, model_path='/data/jfeng/models/mpnet-base-v2', sample_path='/data/luofeng/c2rust/dataset/c2rust_dataset.jsonl', top_k=10, is_cache=1):
    """
    初始化并返回一个可用的 retriever 函数
    """
    if base_dir is None:
        base_dir = os.path.dirname(__file__)

    cache_path = os.path.join(base_dir, "faiss_cache.json")
    if is_cache == 1:
        def retriever(query_code: str):
            return query_with_cache_no_model(query_code, cache_path)
    else:
        index_path = os.path.join(base_dir, "faiss_index_mpnet.index")
        # 加载模型、索引、代码数据
        model = SentenceTransformer(model_path)
        faiss_index = faiss.read_index(index_path)
        code_data = list(jsonlines.open(sample_path))

        def retriever(query_code: str):
            return query_with_cache(query_code, model, faiss_index, code_data, cache_path, top_k=top_k)

    return retriever

def plot_top1_score_distribution(data: list, bins: int = 20, save_path: str = "top1_score_distribution.pdf", normalize: bool = False):
    cache_path = "faiss_cache.json"
    top1_scores = []
    for item in data:
        results = query_with_cache_no_model(item["source_code"], cache_path)
        top1_score = results[0]["score"]
        top1_scores.append(top1_score)

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
    plt.xlabel("Normalized Top-1 Cosine Score" if normalize else "Top-1 Cosine Score")
    plt.ylabel("Number of Samples")
    plt.title("Top-1 Cosine Score Distribution Across Dataset")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    data = load_data()
    plot_top1_score_distribution(data)
    # model_path = '/data/jfeng/models/mpnet-base-v2'
    # sample_path = "/data1/jfeng/c2rust/translation/c_to_rust_Qwen2.5_Coder_32B.jsonl"

    # code_jsonl = "code_only.jsonl"
    # vec_npy = "embeddings_mpnet.npy"
    # index_path = "faiss_index_mpnet.index"
    # # cache_path = "query_cache.jsonl"
    # cache_path = "faiss_cache.json"

    # # 如果还没建立索引，则执行
    # # compute_and_save_embeddings(model_path, sample_path, code_jsonl, vec_npy, index_path)

    # # 查询
    # model = SentenceTransformer(model_path)
    # faiss_index = faiss.read_index(index_path)
    # # code_list, vecs, index = load_code_and_index(code_jsonl, vec_npy, index_path)

    # # 加载代码数据（含 rust_code）
    # code_data = []
    # with jsonlines.open(sample_path, mode='r') as reader:
    #     for obj in reader:
    #         code_data.append(obj)

    # datas = load_data()
    # query_code = datas[1]["source_code"]
    # result = query_with_cache(query_code, model, faiss_index, code_data, cache_path)
    # print(result)  # 可打印或进一步处理
    # for data in datas:
    #     query_code = data["source_code"]
    #     result = query_with_cache(query_code, model, faiss_index, code_data, cache_path)
        # print(result[0])  # 可打印或进一步处理
        # results = find_top_k_faiss(query_code, model, index, code_list, k=10)
        # print(f"Query Code:\n{query_code}\n{'-'*40}")
        # print("Top 10 Similar Codes:")
        # for code, score in results:
        #     print(f"Similarity: {score:.4f}\nCode:\n{code}\n{'-'*40}")
        # input("Press Enter to continue...")
    
