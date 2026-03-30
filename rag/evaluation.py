"""
Evaluation Script
=================
讀取 testset.csv，依序用 RAG 流程回答每道題目，
將結果存至 testset_<mmdd>.csv，新增 Ori Answer 欄位。

使用方式：
  python evaluation.py
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from opencc import OpenCC
from sentence_transformers import SentenceTransformer

from module import (
    load_documents,
    build_or_load_vectorstore,
    load_reranker,
    load_llm,
    build_rag_chain,
    clean_output,
)
from main import (
    CSV_PATH,
    EMBEDDING_MODEL,
    LLM_MODEL_PATH,
    LORA_PATH,
    FAISS_INDEX_PATH,
    RERANKER_MODEL,
    RERANK_CANDIDATES,
    TOP_K,
)

# ─────────────────────────────────────────────
# 評測專用設定
# ─────────────────────────────────────────────

TESTSET_PATH = r"C:\Users\user\Desktop\Mandy\splunk_testset_v2.csv"


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    # 1. 初始化 RAG chain
    docs        = load_documents(CSV_PATH)
    vectorstore = build_or_load_vectorstore(docs, EMBEDDING_MODEL, FAISS_INDEX_PATH)
    reranker    = load_reranker(RERANKER_MODEL)
    llm         = load_llm(LLM_MODEL_PATH, lora_path=LORA_PATH)
    chain       = build_rag_chain(vectorstore, reranker, llm, TOP_K, RERANK_CANDIDATES)

    print(f"[0/5] 載入 Similarity 用 Embedding 模型：{EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    print("    Embedding 模型載入完成")

    cc = OpenCC("s2twp")

    # 2. 讀取 testset（index_col=False 避免 trailing comma 造成欄位偏移）
    df = pd.read_csv(TESTSET_PATH, index_col=False)
    # 移除因 trailing comma 產生的空白 Unnamed 欄位
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    if "Question" not in df.columns:
        raise ValueError(f"找不到 Question 欄位，現有欄位：{df.columns.tolist()}")

    total = len(df)
    results = []
    answers = []
    exact_matches = []
    similarities = []

    print(f"\n共 {total} 道題目，開始評測...\n" + "=" * 50)

    for i, (_, row) in enumerate(df.iterrows(), 1):
        question = str(row["Question"]).strip()
        q_type = str(row.get("Type", "")).strip()
        prompt_question = f"{q_type}：{question}" if q_type else question
        ground_truth = str(row.get("Ground Truth", "")).strip()
        print(f"[{i}/{total}] {prompt_question[:60]}{'...' if len(prompt_question) > 60 else ''}")

        try:
            result = chain.invoke({"question": prompt_question, "retrieval_question": question})
            result = cc.convert(result["result"])
            marker = "回答："
            pos = result.find(marker)
            answer = clean_output(result[pos + len(marker):] if pos != -1 else result)
        except Exception as e:
            result = f"[ERROR] {e}"
            answer = result
            print(f"  ⚠ 發生錯誤：{e}")

        # 計算 Ground Truth 與 Ori Answer 的 cosine similarity
        try:
            vecs = embed_model.encode([ground_truth, answer.strip()], normalize_embeddings=True)
            sim = cosine_similarity(vecs[0], vecs[1])
        except Exception:
            sim = None

        em = int(ground_truth.lower() == answer.strip().lower())

        results.append(result)
        answers.append(answer)
        exact_matches.append(em)
        similarities.append(round(sim, 4) if sim is not None else "")
        print(f"  → 回答完成｜EM={em}｜similarity={sim:.4f}" if sim is not None else f"  → 回答完成｜EM={em}")
        print()

    # 3. 寫回 DataFrame 並儲存
    df["Result"] = results
    df["Answer"] = answers
    df["Exact Match"] = exact_matches
    df["Similarity"] = similarities

    mmdd = datetime.now().strftime("%m%d%H%M")
    stem = os.path.splitext(os.path.basename(TESTSET_PATH))[0]
    base = os.path.dirname(TESTSET_PATH)
    output_path = os.path.join(base, f"{stem}_{mmdd}.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("=" * 50)
    print(f"評測完成！結果已儲存至：{output_path}")


if __name__ == "__main__":
    main()
