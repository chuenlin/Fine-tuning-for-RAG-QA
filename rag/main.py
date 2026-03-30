"""
RAG Service - LangChain + HuggingFace + FAISS
=============================================
架構：
  - Embedding : sentence-transformers（本地）
  - Vector DB : FAISS（純本地）
  - Reranker  : CrossEncoder（BAAI/bge-reranker-v2-m3）
  - LLM       : HuggingFace Pipeline（可直接換 LLaMA Factory 微調模型）
  - 介面      : CLI 問答

使用方式：
  1. 安裝依賴（見底部 requirements）
  2. 把你的 CSV 路徑填入 CSV_PATH
  3. python main.py
"""

from module import (
    load_documents,
    build_or_load_vectorstore,
    load_reranker,
    load_llm,
    build_rag_chain,
    cli_loop,
)

# ─────────────────────────────────────────────
# 設定區（請依需求修改）
# ─────────────────────────────────────────────

# CSV 路徑（欄位名稱為 "chunk"）
CSV_PATH = r"C:\Users\user\Desktop\Mandy\chunk\Splunk\chunk.csv"

# Embedding 模型（中文推薦 shibing624/text2vec-base-chinese）
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"

# LLM 模型路徑（本地資料夾 或 HuggingFace Hub ID）
# 換成 LLaMA Factory 微調模型時，改這裡即可，例如：
#   LLM_MODEL_PATH = "./output/my_finetuned_model"
LLM_MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"

# LoRA adapter 路徑（None 表示只用 base model）
# LORA_PATH = None
LORA_PATH = r"C:\Users\user\Desktop\Mandy\LLaMA-Factory\saves\Qwen2.5-0.5B-Instruct\lora\train_2026-03-23-18-08-00"

# FAISS index 儲存位置（第一次執行後會存檔，之後直接載入）
FAISS_INDEX_PATH = r"C:\Users\user\Desktop\Mandy\faiss_index"

# Reranker 模型（與 BGE embedding 同系列，支援中文）
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# FAISS 初次撈取的候選數量（會再由 reranker 篩選）
RERANK_CANDIDATES = 10

# Rerank 後實際送給 LLM 的 chunk 數量
TOP_K = 3


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────

def main():
    docs = load_documents(CSV_PATH)
    vectorstore = build_or_load_vectorstore(docs, EMBEDDING_MODEL, FAISS_INDEX_PATH)
    reranker = load_reranker(RERANKER_MODEL)
    llm = load_llm(LLM_MODEL_PATH, lora_path=LORA_PATH)
    chain = build_rag_chain(vectorstore, reranker, llm, TOP_K, RERANK_CANDIDATES)
    cli_loop(chain)


if __name__ == "__main__":
    main()
