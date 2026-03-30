#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Splunk 題庫評估報告分析工具

讀取題庫評估產出的 CSV 報告（欄位：ID, Type, Question, Ground Truth, Result, Answer, Exact Match, Similarity），
輸出結構化的 Markdown 分析摘要。

CSV 欄位說明：
  ID            題號
  Type          題型（單選題 / 多選題 / 是非題）
  Question      題目文字
  Ground Truth  正確答案
  Result        RAG prompt（可能含重複內容，會自動截斷）
  Answer        模型回答（可能含重複內容，會自動截斷）
  Exact Match   完全命中（0 或 1）
  Similarity    語意相似度（0.0 ~ 1.0）

使用方法:
  python analyze_report.py <csv_path>
  python analyze_report.py <csv_path> -o report.md
"""

import argparse
import os
import re
from io import StringIO

import pandas as pd

# ── 截斷設定 ──────────────────────────────────────────────────────────────────
# Result / Answer 欄位若出現重複段落（max_token 溢位），截取第一個完整回答區塊
_MAX_ANSWER_CHARS = 800   # 超過此長度時強制截斷，避免報告過長


def _truncate_answer(text: str) -> str:
    """
    截斷因 max_token 重複的 Result / Answer 欄位。
    策略：
      1. 找到第一個「回答：」或「答案：」之後的第一個段落即截斷。
      2. 若找不到關鍵字，則直接截取前 _MAX_ANSWER_CHARS 個字元。
    """
    if not isinstance(text, str):
        return str(text)

    # 嘗試找到「回答：」後的第一段完整回答
    m = re.search(r"(回答：|答案：)(.*?)(?=\n回答：|\n答案：|$)", text, re.DOTALL)
    if m:
        candidate = m.group(0).strip()
        return candidate[:_MAX_ANSWER_CHARS] if len(candidate) > _MAX_ANSWER_CHARS else candidate

    # 退而求其次：直接截斷
    if len(text) > _MAX_ANSWER_CHARS:
        return text[:_MAX_ANSWER_CHARS] + "…（截斷）"
    return text


# ── 相似度門檻 ─────────────────────────────────────────────────────────────────
SIM_LOW = 0.6    # 低相似度警告門檻
SIM_MID = 0.75   # 中等相似度門檻（供統計分組用）


def _safe_cell(text: str, max_len: int = 60) -> str:
    """將文字處理成安全的 markdown table cell：取第一行、跳脫 |、截斷。"""
    first_line = str(text).splitlines()[0].strip() if text else ""
    first_line = first_line[:max_len]
    return first_line.replace("|", "\\|")


def _parse_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _load_df(csv_path: str) -> pd.DataFrame:
    """讀取 CSV 並做前置處理。"""
    df = pd.read_csv(csv_path, encoding="utf-8")

    # 統一欄位名稱（容錯大小寫與前後空白）
    df.columns = [c.strip() for c in df.columns]

    # 截斷可能重複的欄位
    for col in ("Result", "Answer"):
        if col in df.columns:
            df[col] = df[col].apply(_truncate_answer)

    # 數值欄位
    if "Exact Match" in df.columns:
        df["Exact Match"] = _parse_float(df["Exact Match"])
    if "Similarity" in df.columns:
        df["Similarity"] = _parse_float(df["Similarity"])

    return df


# ── 核心分析 ───────────────────────────────────────────────────────────────────

def analyze(csv_path: str) -> str:
    df = _load_df(csv_path)
    total = len(df)
    buf = StringIO()
    w = buf.write

    has_exact = "Exact Match" in df.columns
    has_sim   = "Similarity"  in df.columns
    has_type  = "Type"        in df.columns

    exact_vals = df["Exact Match"].dropna() if has_exact else pd.Series([], dtype=float)
    sim_vals   = df["Similarity"].dropna()  if has_sim   else pd.Series([], dtype=float)

    # ── 標題 ──────────────────────────────────────────────────────────────────
    w("# Splunk 題庫評估報告分析\n\n")

    # ===== 1. 總體概況 ==========================================================
    w("## 1. 總體概況\n\n")
    w("| 項目 | 數值 |\n|------|------|\n")
    w(f"| 總題數 | {total} |\n")

    if has_exact:
        em_count = int(exact_vals.sum())
        w(f"| Exact Match（完全命中）| {em_count} / {total} ({em_count/total*100:.1f}%) |\n")

    if has_sim and len(sim_vals):
        w(f"| 平均相似度 | {sim_vals.mean():.4f} |\n")
        w(f"| 中位數相似度 | {sim_vals.median():.4f} |\n")
        low_sim = (sim_vals < SIM_LOW).sum()
        w(f"| 相似度 < {SIM_LOW} 題數 | {low_sim} ({low_sim/total*100:.1f}%) |\n")
    w("\n")

    # ===== 2. 題型分佈 ==========================================================
    w("## 2. 題型分佈\n\n")
    if has_type and df["Type"].notna().any():
        type_counts = df["Type"].value_counts(dropna=False)
        w("| 題型 | 題數 | 佔比 |\n|------|------|------|\n")
        for t, cnt in type_counts.items():
            label = str(t) if pd.notna(t) else "(未標注)"
            w(f"| {label} | {cnt} | {cnt/total*100:.1f}% |\n")
    else:
        w("（Type 欄位無資料）\n")
    w("\n")

    # ===== 3. 各題型答題表現 ====================================================
    w("## 3. 各題型答題表現\n\n")
    if has_type and df["Type"].notna().any():
        types = sorted(df["Type"].dropna().unique())
        header_cols = ["題型", "題數"]
        if has_exact:
            header_cols.append("Exact Match")
        if has_sim:
            header_cols += ["平均相似度", "中位數相似度", f"相似度<{SIM_LOW}"]
        w("| " + " | ".join(header_cols) + " |\n")
        w("|" + "|".join(["------"] * len(header_cols)) + "|\n")
        for t in types:
            sub = df[df["Type"] == t]
            row_cells = [str(t), str(len(sub))]
            if has_exact:
                em = sub["Exact Match"].dropna()
                row_cells.append(f"{int(em.sum())}/{len(sub)} ({em.mean()*100:.1f}%)" if len(em) else "-")
            if has_sim:
                sv = sub["Similarity"].dropna()
                if len(sv):
                    low = (sv < SIM_LOW).sum()
                    row_cells += [
                        f"{sv.mean():.4f}",
                        f"{sv.median():.4f}",
                        f"{low} ({low/len(sv)*100:.1f}%)",
                    ]
                else:
                    row_cells += ["-", "-", "-"]
            w("| " + " | ".join(row_cells) + " |\n")
    else:
        w("（無題型資料）\n")
    w("\n")

    # ===== 4. 相似度分布 ========================================================
    w("## 4. 相似度分布\n\n")
    if has_sim and len(sim_vals):
        bins = [
            ("0.9 ~ 1.0", sim_vals[(sim_vals >= 0.9) & (sim_vals <= 1.0)]),
            ("0.75 ~ 0.9", sim_vals[(sim_vals >= 0.75) & (sim_vals < 0.9)]),
            (f"{SIM_LOW} ~ 0.75", sim_vals[(sim_vals >= SIM_LOW) & (sim_vals < 0.75)]),
            (f"< {SIM_LOW}", sim_vals[sim_vals < SIM_LOW]),
        ]
        w("| 區間 | 題數 | 佔比 |\n|------|------|------|\n")
        for label, grp in bins:
            w(f"| {label} | {len(grp)} | {len(grp)/total*100:.1f}% |\n")
    else:
        w("（Similarity 欄位無資料）\n")
    w("\n")

    # ===== 5. 問題題目清單 =======================================================
    w("## 5. 問題題目清單\n\n")

    # 5-1: Exact Match = 0（答錯題）
    w("### 5-1. Exact Match = 0（未完全命中）\n\n")
    if has_exact:
        wrong = df[df["Exact Match"] == 0]
        if len(wrong):
            w("| ID | 題型 | 相似度 | 題目 |\n|----|------|--------|------|\n")
            for _, r in wrong.iterrows():
                tid   = r.get("ID", "-")
                ttype = r.get("Type", "-")
                sim   = f"{r['Similarity']:.4f}" if has_sim and pd.notna(r.get("Similarity")) else "-"
                q     = _safe_cell(r.get("Question", ""))
                w(f"| {tid} | {ttype} | {sim} | {q} |\n")
        else:
            w("無（全部完全命中）\n")
    else:
        w("（Exact Match 欄位不存在）\n")
    w("\n")

    # 5-2: 低相似度題目
    w(f"### 5-2. 相似度 < {SIM_LOW}（低相似度）\n\n")
    if has_sim:
        low_df = df[df["Similarity"] < SIM_LOW].dropna(subset=["Similarity"])
        if len(low_df):
            w("| ID | 題型 | Exact Match | 相似度 | 題目 |\n|----|------|------------|--------|------|\n")
            for _, r in low_df.iterrows():
                tid   = r.get("ID", "-")
                ttype = r.get("Type", "-")
                em    = str(int(r["Exact Match"])) if has_exact and pd.notna(r.get("Exact Match")) else "-"
                sim   = f"{r['Similarity']:.4f}"
                q     = _safe_cell(r.get("Question", ""))
                w(f"| {tid} | {ttype} | {em} | {sim} | {q} |\n")
        else:
            w("無\n")
    else:
        w("（Similarity 欄位不存在）\n")
    w("\n")

    # ===== 6. 各題詳細對照 =======================================================
    w("## 6. 各題詳細對照\n\n")
    w("> 此節列出每題的正確答案與模型回答，方便人工核查。\n\n")

    for _, r in df.sort_values("ID").iterrows():
        tid   = r.get("ID", "-")
        ttype = r.get("Type", "-")
        em    = int(r["Exact Match"]) if has_exact and pd.notna(r.get("Exact Match")) else "-"
        sim   = f"{r['Similarity']:.4f}" if has_sim and pd.notna(r.get("Similarity")) else "-"

        w(f"### 題 {tid}　｜　{ttype}　｜　Exact={em}　Sim={sim}\n\n")
        w(f"**題目**\n\n{r.get('Question', '')}\n\n")
        w(f"**正確答案**\n\n{r.get('Ground Truth', '')}\n\n")

        answer_text = _truncate_answer(str(r.get("Answer", "")))
        w(f"**模型回答**\n\n{answer_text}\n\n")
        w("---\n\n")

    return buf.getvalue()


# ── 輸出路徑 ──────────────────────────────────────────────────────────────────

def _default_output_path(csv_path: str) -> str:
    parent   = os.path.dirname(csv_path) or "."
    basename = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir  = os.path.join(parent, "analysis_report")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, basename + ".md")


# ── CLI 入口 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Splunk 題庫評估報告分析工具")
    parser.add_argument("csv", help="評估報告 CSV 路徑")
    parser.add_argument(
        "--output", "-o", default=None,
        help="輸出 Markdown 檔案路徑（預設：同目錄下 analysis_report/ 子目錄）",
    )
    args = parser.parse_args()

    output_path = args.output or _default_output_path(args.csv)
    result      = analyze(args.csv)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"分析報告已輸出至: {output_path}")


if __name__ == "__main__":
    main()