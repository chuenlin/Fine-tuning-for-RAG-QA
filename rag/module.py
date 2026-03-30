import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import CrossEncoder
from peft import PeftModel
import torch
from opencc import OpenCC


# ─────────────────────────────────────────────
# Step 1：載入 CSV -> LangChain Documents
# ─────────────────────────────────────────────

def load_documents(csv_path: str) -> list[Document]:
    print(f"[1/5] 載入 CSV：{csv_path}")
    df = pd.read_csv(csv_path)

    col = None
    for candidate in ["chunk", "chunk_text", "text", "content"]:
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        raise ValueError(f"找不到 chunk 欄位，現有欄位：{df.columns.tolist()}")

    docs = [
        Document(page_content=str(row[col]), metadata={"row_id": idx})
        for idx, row in df.iterrows()
        if pd.notna(row[col]) and str(row[col]).strip()
    ]
    print(f"    共載入 {len(docs)} 筆 chunk")
    return docs


# ─────────────────────────────────────────────
# Step 2：建立或載入 FAISS 向量庫
# ─────────────────────────────────────────────

def build_or_load_vectorstore(docs: list[Document], embedding_model: str, index_path: str) -> FAISS:
    print(f"[2/5] Embedding 模型：{embedding_model}")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if os.path.exists(index_path):
        print(f"    發現已存在的 FAISS index，直接載入：{index_path}")
        vectorstore = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        print(f"    第一次建立 FAISS index（共 {len(docs)} 筆），請稍候...")
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(index_path)
        print(f"    FAISS index 已儲存至：{index_path}")

    return vectorstore


# ─────────────────────────────────────────────
# Step 3：載入 Reranker（CrossEncoder）
# ─────────────────────────────────────────────

def load_reranker(model_name: str) -> CrossEncoder:
    print(f"[3/5] 載入 Reranker：{model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoder(model_name, device=device)
    print("    Reranker 載入完成")
    return reranker


def rerank_docs(reranker: CrossEncoder, question: str, docs: list[Document], top_n: int) -> list[Document]:
    if not docs:
        return docs
    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


# ─────────────────────────────────────────────
# Step 4：載入 LLM（HuggingFace Pipeline）
# ─────────────────────────────────────────────

def load_llm(model_path: str, lora_path: str = None) -> HuggingFacePipeline:
    print(f"[4/5] 載入 LLM：{model_path}")
    device = 0 if torch.cuda.is_available() else -1
    print(f"    使用裝置：{'GPU' if device == 0 else 'CPU'}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if lora_path:
        print(f"    套用 LoRA adapter：{lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()  # 合併 LoRA 權重，推論速度更快
        print("    LoRA 合併完成")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.1,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    print("    LLM 載入完成")
    return llm


# ─────────────────────────────────────────────
# Step 5：建立 RAG Chain（含 Rerank）
# ─────────────────────────────────────────────

def build_rag_chain(vectorstore: FAISS, reranker: CrossEncoder, llm: HuggingFacePipeline, top_k: int, rerank_candidates: int):
    print(f"[5/5] 建立 RAG Chain（candidates={rerank_candidates} → rerank → top_k={top_k}）")

    prompt_template = """你是一個專業的問答助理。請根據以下提供的參考資料，回答使用者的問題。
如果是選擇題或是非題，請直接給出正確完整選項（例如 (a) xxx）或是「正確/錯誤」，不需要說明任何理由。多選題請用換行分隔。
如果參考資料中沒有相關資訊，請直接說明「根據現有資料無法回答」，不要自行編造答案。

參考資料：
{context}

問題：{question}

回答："""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": rerank_candidates})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def retrieve_and_rerank(x):
        retrieval_question = x.get("retrieval_question") or x["question"]
        candidates = retriever.invoke(retrieval_question)
        return rerank_docs(reranker, retrieval_question, candidates, top_k)

    chain = (
        RunnablePassthrough.assign(
            source_documents=retrieve_and_rerank,
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(x["source_documents"]),
        )
        | RunnablePassthrough.assign(
            result=(prompt | llm | StrOutputParser())
        )
    )

    print("    RAG Chain 建立完成\n")
    return chain


# ─────────────────────────────────────────────
# 後處理
# ─────────────────────────────────────────────

def clean_output(text: str) -> str:
    text = text.strip()

    # 模型第一句就是「無法回答」時直接回傳，避免後面重複生成的內容干擾
    first_line = text.splitlines()[0].strip() if text else ""
    if "根據現有資料無法回答" in first_line:
        return first_line

    # 模型重複生成問題/回答時，截斷至第一個重複標記前
    cutoff_markers = ["問題：", "\n問題", "根據現有資料無法回答","根據上述"]
    positions = [text.find(m) for m in cutoff_markers if text.find(m) != -1]

    if positions:
        return text[:min(positions)].strip()

    return text


# ─────────────────────────────────────────────
# CLI 問答介面
# ─────────────────────────────────────────────

def cli_loop(chain):
    print("=" * 50)
    print("RAG 問答服務已啟動（輸入 'exit' 或 'quit' 離開）")
    print("=" * 50)

    while True:
        try:
            question = input("\n請輸入問題：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n已離開。")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("已離開。")
            break

        print("\n思考中...")
        result = chain.invoke({"question": question})

        print("\n【回答】")
        cc = OpenCC("s2twp")
        print(clean_output(cc.convert(result["result"])))

        print("\n【參考來源 chunk】")
        for i, doc in enumerate(result["source_documents"], 1):
            preview = doc.page_content[:100].replace("\n", " ")
            print(f"  [{i}] row_id={doc.metadata.get('row_id', '?')} | {preview}...")
