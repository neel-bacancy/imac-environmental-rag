"""Evaluate the iMac RAG on hand-checked questions.

    python -m eval.run_eval               # reranker as set in app/config.py
    python -m eval.run_eval --compare     # reranker off and on, side by side

A question passes when:
  * retrieval: at least one of the top sources is on an expected page (skipped for refusal questions), and
  * answer:    every must_contain pattern matches, no forbid pattern matches, and refusals refuse.
Patterns are regular expressions, so "aluminum|aluminium" and "58\\s?%" match the printed forms.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime

from app.answer import answer, load_retriever
from app.config import ROOT, get_settings
from app.llm import Gemini

QUESTIONS = ROOT / "eval" / "sample_questions.json"
RESULTS = ROOT / "eval" / "results"


def retrieval_hit(question: dict, sources) -> bool:
    if not question.get("expected_pages"):
        return True
    return any(c.page in question["expected_pages"] for c in sources)


def answer_ok(question: dict, text: str, refused: bool) -> bool:
    if question.get("refuse"):
        return refused
    if refused:
        return False
    return all(re.search(p, text, re.I) for p in question["must_contain"]) and not any(
        re.search(p, text, re.I) for p in question["forbid"]
    )


def run(rerank: bool, retriever, gemini, settings, questions) -> dict:
    rows = []
    for q in questions:
        result = answer(q["question"], retriever, gemini, settings, rerank=rerank)
        hit = retrieval_hit(q, result.sources)
        ok = answer_ok(q, result.text, result.refused)
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "search_query": result.query,
                "sources": [f"{c.kind} p{c.page} ({(c.section or '')[:40]})" for c in result.sources],
                "retrieval_hit": hit,
                "answer_pass": ok,
                "passed": hit and ok,
                "answer": result.text,
            }
        )
        status = "PASS" if hit and ok else "FAIL"
        print(f"[{status}] {q['id']} retrieval={'ok' if hit else 'MISS'} answer={'ok' if ok else 'BAD'} | {q['question'][:70]}")
    total = len(rows)
    return {
        "rerank": rerank,
        "passed": sum(r["passed"] for r in rows),
        "total": total,
        "retrieval_hit_rate": sum(r["retrieval_hit"] for r in rows) / total,
        "answer_pass_rate": sum(r["answer_pass"] for r in rows) / total,
        "rows": rows,
    }


def write_report(runs: list[dict], stamp: str):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{stamp}.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"# iMac evaluation {stamp}", "", "| Reranker | Passed | Retrieval hit | Answer pass |", "|---|---|---|---|"]
    for r in runs:
        lines.append(f"| {'on' if r['rerank'] else 'off'} | {r['passed']}/{r['total']} | {r['retrieval_hit_rate']:.0%} | {r['answer_pass_rate']:.0%} |")
    for r in runs:
        lines += ["", f"## Reranker {'on' if r['rerank'] else 'off'}", ""]
        for row in r["rows"]:
            lines += [
                f"### {row['id']}: {row['question']}",
                f"- Result: **{'PASS' if row['passed'] else 'FAIL'}** (retrieval {'ok' if row['retrieval_hit'] else 'miss'}, answer {'ok' if row['answer_pass'] else 'bad'})",
                f"- Search query: {row['search_query']}",
                f"- Sources: {'; '.join(row['sources'])}",
                "",
                row["answer"],
                "",
            ]
    path = RESULTS / f"{stamp}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    gemini = Gemini(settings)
    retriever = load_retriever(settings, gemini)
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))

    modes = [False, True] if args.compare else [settings.use_reranker]
    runs = []
    for mode in modes:
        print(f"\n== reranker {'on' if mode else 'off'} ==")
        runs.append(run(mode, retriever, gemini, settings, questions))
    path = write_report(runs, datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f"\nreport: {path}")


if __name__ == "__main__":
    main()
