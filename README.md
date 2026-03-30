# Fine-tuning for RAG QA
## Evaluating Answer Accuracy and Format Controllability

> **核心問題**：指令微調能讓 LLM 在 RAG 場景下的選擇題回答更準確、更可控嗎？怎麼驗證？

---

## 1. 專案背景與動機

將 LLM 整合進 RAG 系統時，選擇題場景有兩個常見問題：

- **格式失控**：模型回答時加上解釋、重述題目，導致答案無法被程式正確解析
- **多選題只答一個**：題目要求選多個選項，模型卻只回答一項

這個實驗想驗證的假設是：**指令微調（Instruction Fine-tuning）能同時改善「答案正確率」和「輸出格式可控性」**。

知識來源為 Splunk 搜尋操作教材，題型涵蓋單選、多選、是非三種，透過 RAG Pipeline 搭配微調前後的模型進行對比評估。

---

## 2. 架構

```
                    ┌─────────────────────────────────────┐
                    │           RAG Pipeline               │
                    │                                      │
  chunk.csv ───────►│  FAISS (BGE embedding)               │
                    │      ↓                               │
  Question ────────►│  CrossEncoder Reranker               │
                    │      ↓                               │
                    │  Qwen2.5-0.5B-Instruct               │
                    │  (base / + LoRA adapter)             │
                    │      ↓                               │
                    │  Answer                              │
                    └─────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │         Evaluation Flow              │
                    │                                      │
  splunk_testset ──►│  RAG 回答                            │
                    │      ↓                               │
                    │  Exact Match   ← 答案對不對？        │
                    │  Cosine Sim    ← 語意對不對？         │
                    │      ↓                               │
                    │  Result CSV                          │
                    └─────────────────────────────────────┘
```

**使用的模型與工具：**

| 元件 | 選擇 |
|------|------|
| Embedding | `BAAI/bge-large-zh-v1.5` |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| LLM | `Qwen/Qwen2.5-0.5B-Instruct` |
| 微調框架 | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) |
| RAG 框架 | LangChain + FAISS |

---

## 3. 評估設計說明

### 為什麼用 Exact Match + Cosine Similarity 雙指標？

單一指標無法區分「答錯」和「答對但格式不符」這兩種失敗模式：

- **Exact Match（EM）**：判斷答案是否與 Ground Truth 完全一致，反映最終可用性
- **Cosine Similarity**：計算模型輸出與 Ground Truth 的語意向量距離，反映模型「是否其實知道答案」

兩個指標組合才能診斷問題：若 Cosine 高但 EM 低 → 語意正確但格式出錯；若兩者都低 → 真的答錯了。

### 為什麼不評估 Retrieval 層？

這個實驗的目標是比較**同一 RAG 環境下**，微調前後模型的差異。Retrieval 品質兩組完全相同，不是這次想觀察的變因，因此刻意簡化，不加入 Retrieval 層的評估指標。

### 題目與 Ground Truth 的品質保證

- 知識來源為 Splunk 教育訓練教材，chunk 與題目的對應關係明確
- 題目與答案經過人工審查與多輪 AI 輔助修正，確保 GT 本身正確且格式一致

---

## 4. 實驗結果

模型：`Qwen2.5-0.5B-Instruct`，微調方式：LoRA（使用 LLaMA-Factory）

測試集：`data/splunk_testset.csv`（15 題：單選 6、多選 4、是非 5）

| 題型 | N | 微調前 EM | 微調前 Cosine | 微調後 EM | 微調後 Cosine |
|------|---|-----------|---------------|-----------|---------------|
| 單選題 | 6 | 16.7% | 0.658 | 50.0% | 0.860 |
| 多選題 | 4 | 0.0% | 0.553 | 0.0% | 0.909 |
| 是非題 | 5 | 20.0% | 0.601 | 20.0% | 0.656 |
| **Overall** | **15** | **13.3%** | **0.611** | **26.7%** | **0.805** |

### 結果解讀

- **單選題**：EM 從 16.7% 大幅提升至 50%，Cosine 從 0.66 升至 0.86，微調效果最明顯
- **多選題**：EM 維持 0%，但 Cosine 從 0.55 大幅上升至 0.91，代表模型語意上已理解正確答案，仍是**格式問題**（換行分隔未完全學會）
- **是非題**：EM 持平，Cosine 僅微幅提升，微調對此題型改善有限，推測與訓練資料中是非題比例較少有關

---

## 5. 踩過的坑

### Ground Truth 品質問題

評估初期發現部分題目 EM 異常偏低，但模型答案看起來正確。排查後發現問題出在 GT 本身：部分題目的 GT 格式不一致（例如「正確」vs「是」、選項文字有多餘空白），導致 EM 算法誤判為錯誤。

解決方式：統一 GT 格式標準，並針對每道題做人工逐一確認，加入 OpenCC 繁簡轉換避免字型差異造成的誤判。

### 微調資料的格式設計影響輸出行為

最初的微調資料 output 欄位格式不夠嚴格（例如多選題用頓號分隔而非換行），導致微調後的模型輸出格式與評估腳本預期不一致。

調整後將 output 格式對齊評估腳本的 `clean_output()` 函式，微調後的輸出格式才穩定符合預期。

---

## 6. 檔案結構

```
.
├── data/
│   ├── chunk.csv                   # 知識庫（由 Splunk 教材切分）
│   ├── splunk_testset.csv          # 測試題目（15 題，含 Ground Truth）
│   └── splunk_finetune_data.json   # LoRA 微調訓練資料（Alpaca 格式）
│
├── rag/
│   ├── main.py                     # RAG pipeline 進入點（含設定區）
│   ├── module.py                   # 核心模組：embedding / rerank / LLM / chain
│   ├── evaluation.py               # 自動評估腳本（EM + Cosine Similarity）
│   ├── analyze_report.py           # 結果分析輸出 Markdown 報告
│   └── requirements.txt            # Python 依賴
│
├── chunk/
│   └── preprocess.py               # PDF → 語意切分 → chunk.csv
│
└── eval_result/
    ├── eval_result.xlsx                              # 彙整結果
    ├── eval_v2_03251950_0.5B.csv                    # 微調前評測結果
    └── eval_v2_03260034_0.5B_lora03231808.csv       # 微調後評測結果
```

---

## 7. 快速上手

### 環境安裝

```bash
pip install -r rag/requirements.txt
```

### 設定路徑

編輯 [rag/main.py](rag/main.py) 的設定區，填入本地路徑：

```python
CSV_PATH         = "data/chunk.csv"
LLM_MODEL_PATH   = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH        = None  # 套用 LoRA 時填入 checkpoint 路徑
FAISS_INDEX_PATH = "faiss_index"
```

### 執行評估

```bash
python rag/evaluation.py
```

結果會存至 `eval_result/` 目錄下，可用 `analyze_report.py` 產出 Markdown 報告：

```bash
python rag/analyze_report.py eval_result/<filename>.csv
```

### 重新切分知識庫

若要從 PDF 重新生成 `chunk.csv`，將 PDF 放入 `chunk/Splunk/` 後執行：

```bash
python chunk/preprocess.py
```

---

## 8. 微調說明

微調使用 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，訓練資料格式為 Alpaca（`instruction` / `input` / `output`）。

訓練資料位於 `data/splunk_finetune_data.json`，每筆資料包含：
- `instruction`：角色設定與輸出格式要求
- `input`：RAG 的 context + 問題
- `output`：正確答案（格式與評估腳本對齊）
