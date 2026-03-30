# Fine-tuning for RAG QA
### Evaluating Answer Accuracy and Format Controllability

> **核心問題**：指令微調能讓 LLM 在 RAG 場景下的選擇題回答更準確、更可控嗎？怎麼驗證？

---

## 1. 專案背景與動機

LLM 在 RAG（Retrieval-Augmented Generation）問答場景中，即使能從 chunk 找到正確資訊，也常出現兩類問題：

1. **格式失控**：答案正確但輸出格式不符（單選只輸出代號、多選答案未換行分隔），導致自動評估誤判為錯誤
2. **輸出不穩定**：模型在答完答案後繼續生成無關內容，或重複輸出

本專案驗證的假設：**指令微調（SFT + LoRA）能同時改善「輸出格式可控性」和「答案正確率」**，並透過雙指標設計區分這兩種改善。

知識來源為 Splunk 搜尋操作教材，題型涵蓋單選、多選、是非三種。選擇此來源的原因是 chunk 與題目的對應關係明確，能確保題目與 GT 的品質，同時結合實際工作場景的領域知識。

---

## 2. 系統架構

```
RAG Pipeline
────────────
原始文件（Splunk 教材 PDF/投影片）
    ↓ 切割
FAISS 向量索引
    ↓ 相似度搜尋（top-k=3）
CrossEncoder Reranker
    ↓ 精排
Chunk → Prompt
    ↓
Qwen2.5-0.5B-Instruct + LoRA（HuggingFace Pipeline）
    ↓
模型輸出

評估流程
────────
測試題目（15題，含人工審查 GT）
    ↓
Exact Match + Cosine Similarity 雙指標
    ↓
微調前後對比分析
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

### 為什麼用雙指標？

| 指標 | 衡量的問題 | 限制 |
|---|---|---|
| Exact Match（EM）| 答案是否完全正確 | 對格式敏感，格式錯就是 0 |
| Cosine Similarity | 答案語意是否接近 | 能反映「答對但格式不符」的情況 |

**兩個指標的差距就是資訊：**
- EM=0 但 Similarity 高 → 模型知道答案，但格式跑掉
- EM=0 且 Similarity 低 → 模型真的答錯了

這個組合讓我們能區分「格式問題」和「推理問題」，而不是把兩者混在一起計算。

### 為什麼不評估 Retrieval 層？

本實驗的目的是單純比較**微調前後**的差異，使用固定的 RAG 設定（top-k=3, CrossEncoder reranker）作為控制變數。如果同時評估 Retrieval，變數太多反而無法歸因。

### GT 品質保證

- 題目來源：Splunk 教育訓練課程作業，chunk 與題目對應關係明確
- GT 建立：由人工撰寫並逐題審查
- 多輪 AI 輔助品質檢查：確認問題、chunk、GT 三者相互符合
- 加入 OpenCC 繁簡轉換，避免字型差異造成 EM 誤判

---

## 4. 實驗結果

### 整體指標（15 題）

| 模型狀態 | Exact Match | Cosine Similarity |
|---|---|---|
| 無 LoRA（baseline）| 2/15（13.3%）| 0.611 |
| **+ LoRA 微調** | **4/15（26.7%）** | **0.805** |

### 各題型分析

| 題型 | 無 LoRA EM | LoRA EM | 無 LoRA Sim | LoRA Sim |
|---|---|---|---|---|
| 單選題（6題）| 1/6（16.7%）| **3/6（50.0%）** | 0.658 | **0.860** |
| 是非題（5題）| 1/5（20.0%）| 1/5（20.0%）| 0.601 | 0.656 |
| 多選題（4題）| 0/4（0%）| 0/4（0%）| 0.553 | **0.909** |

### 解讀

**單選題改善最顯著**：EM 從 16.7% 提升至 50%，格式控制效果明確。

**多選題的關鍵發現**：EM 維持 0%（完全沒有答對），但 Similarity 從 0.553 大幅提升至 0.909。這說明微調後模型的答題方向正確，但有漏選或多選。這個差距正是雙指標設計的意義所在，如果只看 EM，會誤以為「多選題完全沒有改善」。

**是非題幾乎無改善**：EM 和 Similarity 均無明顯提升，根因分析指向 0.5B 模型的推理能力上限。這類題目需要「比對 chunk 與題目描述是否矛盾」的能力，是純格式調整無法解決的問題。

---

## 5. 踩過的坑

### 坑一：GT 格式不一致導致 EM 誤判

評估初期發現部分題目 EM 異常偏低，但模型答案看起來正確。排查後發現問題出在 GT 本身，部分題目的 GT 格式不一致（例如選項文字有多餘空白、繁簡字型差異），導致 EM 算法誤判為錯誤。

解法：統一 GT 格式標準，逐題人工確認，並加入 OpenCC 繁簡轉換處理字型差異。

### 坑二：訓練資料格式不一致導致輸出退化

初期多選題訓練資料的答案格式混用逗號分隔和換行分隔，微調後模型的多選輸出格式變得不穩定。發現方式：觀察到 Similarity 在某版本比上一版低，逐筆檢查輸出才發現格式退化。

解法：統一所有多選題 output 格式為換行分隔，並在訓練前用程式自動檢查格式一致性，同時將 output 格式對齊評估腳本的 `clean_output()` 函式。

### 坑三：訓練資料題型比例失衡導致模型偏差

後期為了改善是非題表現，集中補充大量「答案為錯誤」的是非題，使是非題佔比從 17% 升至 43%。結果模型開始把單選題也當成是非題來回答，輸出「正確」或「錯誤」。

解法：維持題型比例均衡（最終版本單選 30%、是非 31%、多選 39%），是非題內正確與錯誤也維持接近 1:1。

### 坑四：Loss 低不代表結果好

1.5B 模型在相同資料量下 loss 降到 0.07~0.09，看起來比 0.5B 的 0.196 優秀，但實際推理結果更差（格式混亂、重複輸出）。也就是在小資料集上的典型的 overfitting 問題。

---

## 6. 技術細節

| 項目 | 設定 |
|---|---|
| 基底模型 | Qwen2.5-0.5B-Instruct |
| 微調方式 | SFT + LoRA（rank=8, alpha=16, dropout=0.05）|
| 訓練框架 | LLaMA-Factory |
| 訓練資料 | 80 筆（單選 30%、是非 31%、多選 39%）|
| 訓練平台 | Google Colab（T4 GPU）|
| RAG 框架 | LangChain + FAISS + CrossEncoder |
| 評估指標 | Exact Match + Cosine Similarity（sentence-transformers）|

---

## 7. 未來方向

### 短期：改善 RAG Chunk 品質

目前是非題表現無法透過微調改善，根因分析顯示 chunk 本身缺乏「否定判斷的依據」，chunk 描述了正確知識，但沒有明確指出哪些描述是錯的。因此下一步應在 chunk 切割策略上下手，確保每個 chunk 包含足夠的對比資訊，讓模型能做出「與題目描述矛盾即錯誤」的判斷。

### 中期：擴充訓練資料 + 換用更大的模型

本實驗使用 80 筆訓練資料，足以讓 0.5B 模型學會輸出格式，但不足以讓更大的模型（1.5B+）發揮推理優勢。實驗過程中嘗試 1.5B 模型均因資料量不足而 overfit（loss < 0.08），輸出格式反而比 0.5B 更差。估計需要至少 300~500 筆資料，才能讓 1.5B 的推理能力在是非題比對矛盾和多選題邊界判斷上體現出來。

### 長期：更細緻的評估框架

目前使用 Exact Match + Cosine Similarity 評估整體問答品質，但無法拆解「Retrieval 撈錯」和「Generation 推理錯」兩種失敗來源。引入 Retrieval 層評估（如 Context Recall、Answer Faithfulness）可以讓問題定位更精準，進一步指引改善方向。

---

## 8. 檔案結構

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
    ├── eval_v2_03260034_0.5B_lora03231808.csv       # 微調後評測結果
    └── analysis_report/
        ├── eval_v2_03251950_0.5B.md                 # 微調前逐題分析報告
        └── eval_v2_03260034_0.5B_lora03231808.md    # 微調後逐題分析報告
```

---

## 9. 環境建置與執行

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

結果會存至 `eval_result/` 目錄下，可用 `analyze_report.py` 產出逐題 Markdown 分析報告：

```bash
python rag/analyze_report.py eval_result/<filename>.csv
```

### 重新切分知識庫

若要從 PDF 重新生成 `chunk.csv`，將 PDF 放入 `chunk/Splunk/` 後執行：

```bash
python chunk/preprocess.py
```

---

## 10. 微調說明

微調使用 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，訓練資料格式為 Alpaca（`instruction` / `input` / `output`）。

訓練資料位於 `data/splunk_finetune_data.json`，每筆資料包含：
- `instruction`：角色設定與輸出格式要求
- `input`：RAG 的 context + 問題
- `output`：正確答案（格式與評估腳本對齊）
