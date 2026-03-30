import os
import re
import numpy as np
import pandas as pd
import torch
import fitz  # pymupdf
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 設定區
# =========================

EMBEDDINGS_MODEL = "BAAI/bge-large-zh-v1.5"
FAISS_OUTPUT     = r"C:\Users\user\Desktop\Mandy\faiss_index"
CHUNK_CSV_OUTPUT = r"C:\Users\user\Desktop\Mandy\chunk\Splunk\chunk.csv"

# 處理的檔案清單（可自行調整）
FILES = [
    {
        "path":      r"C:\Users\user\Desktop\Mandy\chunk\Splunk\Splunk搜尋(一).pdf",
        "threshold": 0.75,
        "max_chars": 200,   #PCB:400 #splunk:250
    },
    {
        "path":      r"C:\Users\user\Desktop\Mandy\chunk\Splunk\Splunk搜尋(二).pdf",
        "threshold": 0.75,
        "max_chars": 200,   #PCB:400 #splunk:250
    },
    {
        "path":      r"C:\Users\user\Desktop\Mandy\chunk\Splunk\Splunk搜尋(三).pdf",
        "threshold": 0.75,
        "max_chars": 200,   #PCB:400 #splunk:250
    }
]

# =========================
# 單一檔案處理
# =========================

def preprocess_file(embeddings, text_file_path, threshold=0.75, max_chars=250):
    print(f"\n  讀取：{os.path.basename(text_file_path)}")

    # --- 讀取內容 ---
    ext = os.path.splitext(text_file_path)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(text_file_path)
        pages_text = []
        for page in doc:
            blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
            # block_type 0 = text, 1 = image
            text_blocks = [b[4] for b in blocks if b[6] == 0]
            pages_text.append("\n".join(text_blocks))
        doc.close()
        content = "\n".join(pages_text)
    else:
        with open(text_file_path, "r", encoding="utf-8-sig") as f:
            content = f.read()

    # --- 清除雜訊 ---
    lines = content.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 頁首：「電路板與載板術語手冊」（含或不含頁碼）
        if re.match(r'^電路板與載板術語手冊', line):
            continue
        # 頁尾：含 .indb 的印刷檔資訊
        if '.indb' in line:
            continue
        # 過短的行（通常是圖說殘片或頁碼）
        if len(line) < 4:
            continue
        # 圖說前綴「S 」開頭的行（圖片標題）
        # if re.match(r'^S\s{1,3}\S', line):
        #     continue
        # 圖片標籤：中文字比例過低（大量英文/符號夾雜，通常是圖說標籤）
        # chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', line))
        # if len(line) > 10 and chinese_chars / len(line) < 0.1:
        #     continue
        cleaned.append(line)
    content = "\n".join(cleaned)

    # --- 語意切分 ---
    # 1. Sentence-level split (coarse)
    sentences = re.split(r"(?:。|\n)", content)
    sentences = [s.strip() for s in sentences if s.strip()]

    # 2. Sentence embeddings
    sentence_embeddings = embeddings.embed_documents(sentences)

    # 3. Semantic boundary detection
    semantic_chunks = []
    current_chunk = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_similarity(
            [sentence_embeddings[i - 1]],
            [sentence_embeddings[i]]
        )[0][0]
        if sim < threshold:
            semantic_chunks.append("".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    if current_chunk:
        semantic_chunks.append("".join(current_chunk))

    print(f"  語意切分後 chunks 數量: {len(semantic_chunks)}")

    # --- 長度控制 ---
    final_chunks = []
    buffer = ""
    for chunk in semantic_chunks:
        if len(buffer) + len(chunk) <= max_chars:
            buffer += chunk
        else:
            final_chunks.append(buffer)
            buffer = chunk
    if buffer:
        final_chunks.append(buffer)

    print(f"  最終 chunks 數量: {len(final_chunks)}")
    return final_chunks


# =========================
# 主程式
# =========================

print(f"載入 embedding 模型: {EMBEDDINGS_MODEL}")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDINGS_MODEL,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

all_chunks = []
for file_cfg in FILES:
    chunks = preprocess_file(
        embeddings=embeddings,
        text_file_path=file_cfg["path"],
        threshold=file_cfg.get("threshold", 0.75),
        max_chars=file_cfg.get("max_chars", 250),
    )
    all_chunks.extend(chunks)

print(f"\n所有檔案合計 chunks 數量：{len(all_chunks)}")

# chunk vector
print("建立 FAISS 向量資料庫...")
vectorstore = FAISS.from_texts(texts=all_chunks, embedding=embeddings)
vectorstore.save_local(FAISS_OUTPUT)
print(f"FAISS index 已儲存至：{FAISS_OUTPUT}")

# 存 chunk 原文
pd.DataFrame({"chunk": all_chunks}).to_csv(CHUNK_CSV_OUTPUT, encoding="utf-8-sig", index=False)
print(f"chunk.csv 已儲存至：{CHUNK_CSV_OUTPUT}")
