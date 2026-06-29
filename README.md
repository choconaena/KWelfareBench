# KWelfareBench

**A cell-level eligibility benchmark and conversational architecture for Korean welfare recommendation.**

Companion repository for the paper *"Closing the Welfare Outreach Gap: A Conversational Architecture and Cell-Level Eligibility Benchmark for Korean Welfare Recommendation."*

> **Accepted at the ICML 2026 AI for Good Workshop (AI for Public Institutions track).**

**Authors:** Byeongmin Kang¹, Minwoo Han²†, Junhak Lee³†, Minsu Kim²†, Jihie Kim¹ (corresponding)
¹ Department of Computer Science and Artificial Intelligence, Dongguk University, Seoul, Republic of Korea
² INNO-HI Inc., Republic of Korea  ³ RaonSecure Co., Ltd., Republic of Korea
† Equal contribution (co-second authors). Correspondence: `jihie.kim@dgu.edu`

---

## What's released

**KWelfareBench** is the benchmark introduced in §3 of the paper. It contains nine artefacts (A1–A9):

| ID | Artefact | Format | Size |
|----|----------|--------|------|
| A1 | Raw policies | JSON | 4,937 policies |
| A2 | Regional standard | JSON | 17 sido / 226 sigungu |
| A3 | Policy tags | JSON | 4,937 × 72 |
| A4 | Synthetic personas | JSON | 180 personas |
| A5 | Persona tags | JSON | 180 × 72 |
| A6 | Query pool | JSON | 33 sub-topics, 66 queries |
| A7 | GT-1 eligibility | JSON | 888,660 cells |
| A8 | GT-2 intent (graded) | JSONL | 325,842 cells |
| A9 | GT-3 intersection (×3 region) | JSON | 11,880 × 3 |

All artefacts are released under **KOGL Type 1**, the same licence as the upstream Bokjiro source.

---

## Repository layout

```
.
├── paper/                          ICML 2026 camera-ready LaTeX sources
│   ├── paper_main.tex              Master source (compile with pdflatex)
│   ├── paper_main.pdf              Compiled camera-ready PDF
│   ├── refs.bib                    Bibliography
│   ├── figures/                    All 7 paper figures + sources
│   ├── icml2026.sty / .bst         Official ICML 2026 style files
│   └── *.sty                       Supporting LaTeX packages
│
├── data/                           Released benchmark
│   ├── policies.json               A1: 4,937 Bokjiro policies
│   ├── regions.json                A2: 17 sido / 226 sigungu standard
│   ├── policy_tags_labels.json     A3: 4,937 × 72 policy tags
│   ├── personas_v3.json            A4: 180 synthetic personas
│   ├── persona_schema_v3.json      A5: persona tag schema
│   ├── query_pool_v3.json          A6: 33 sub-topics × 2 phrasings
│   ├── ground_truth_v4*.json       A7: GT-1 eligibility (rule-based) + stats
│   ├── gt2_openai_final.jsonl      A8: GT-2 graded intent (primary judge)
│   ├── gt2_gemini_final.jsonl      A8: GT-2 graded intent (cross-LLM judge)
│   ├── gt3_*.json                  A9: GT-3 intersection (×3 region variants) + stats
│   └── graph/eligibility_graph.pkl Tag-overlap graph (B4)
│
├── experiments/                    Reproducible experiment outputs
│   ├── gt2_baselines_results.json     §4 GT-2 NDCG@10 / P@10 / R@10
│   ├── gt3_baselines_results.json     §4 GT-3 strict (B1–B4)
│   ├── fair_compare_gt3.json          B5+rule / B6 matched-prefilter
│   ├── baseline_ci_results.json       95% bootstrap CIs + Wilcoxon p-values
│   ├── latency_results.json           §4.4 per-query latency
│   ├── gt2_kappa_analysis.json        Cross-LLM Cohen κ
│   ├── gt2_human_review_summary.json  Human spot-check audit
│   ├── b6_rerank/                     B6 dense/Gemini rerank scores
│   ├── r1_ablation/ , r1_ablation_multi/  §4 R1-violation ablations
│   └── r13_phase3/                    §5 conversational architecture eval
│
├── scripts/                        Reproducibility code
│   ├── retrievers/                 B1–B4 baseline implementations
│   ├── build_personas_v3.py        Persona generator (A4)
│   ├── build_query_pool.py         Query pool curation (A6)
│   ├── compute_gt1_v3.py           GT-1 deterministic eligibility (A7)
│   ├── build_gt3.py                GT-3 matrix builder (A9)
│   ├── run_gt2_baselines.py        §4 GT-2 evaluation
│   ├── run_gt2_gemini_sync.py      GT-2 grading (cross-LLM)
│   ├── run_b6_llm_rerank.py        B6 rerank
│   ├── run_r1_multi_prefix.py      §4 R1-violation multi-prefix
│   └── bootstrap_ci.py             Bootstrap CIs + paired Wilcoxon
│
├── LICENSE                         KOGL Type 1 (data) + MIT (code)
└── README.md                       This file
```

---

## Reproducibility quick-start

**Environment**: Python 3.10+, single CPU is sufficient for retrieval baselines; GPU recommended for ko-SRoBERTa and cross-encoder rerankers.

```bash
pip install rank-bm25 sentence-transformers numpy pandas tqdm \
            openai google-genai scikit-learn

python scripts/run_gt2_baselines.py     # GT-2 intent-only (Table 2)
python scripts/run_b6_llm_rerank.py      # B6 rule+dense rerank (GEMINI_API_KEY for rerank judge)
python scripts/run_r1_multi_prefix.py    # R1-violation ablation (Figure, §3)
python scripts/bootstrap_ci.py           # bootstrap CIs + Wilcoxon
```

API keys are read from environment variables / a local `.env` (never committed):
`OPENAI_API_KEY`, `GEMINI_API_KEY`. Personas are fully synthetic (no PII).

LLM judges / models used: OpenAI `gpt-5.4-nano` (primary GT-2 judge), Google `gemini-2.5-flash-lite` (cross-LLM check, B5 rerank), `jhgan/ko-sroberta-multitask` (B2 dense retrieval).

---

## Compiling the paper

```bash
cd paper
pdflatex paper_main.tex
bibtex   paper_main
pdflatex paper_main.tex
pdflatex paper_main.tex
```

---

## What this repository is *not*

- **A deployment artefact.** The kiosk frontend and backend are not included; this repo is scoped to the paper's reproducibility and benchmark release.
- **A controlled-study artefact.** The conversational architecture is evaluated in simulation using GT-3; no real user logs are released.
- **A SOTA recommendation system.** §5 scopes the trained selector explicitly as a *deployment-feasibility demonstration*: ε-greedy(IG) matches it on accuracy; the differentiator is inference cost (~1 ms vs. ~60 ms per turn).

---

## Citation

```bibtex
@inproceedings{kang2026kwelfarebench,
  title     = {Closing the Welfare Outreach Gap: A Conversational Architecture
               and Cell-Level Eligibility Benchmark for Korean Welfare Recommendation},
  author    = {Kang, Byeongmin and Han, Minwoo and Lee, Junhak and Kim, Minsu and Kim, Jihie},
  booktitle = {ICML 2026 AI for Good Workshop},
  year      = {2026}
}
```

---

## License

| Asset | Licence |
|-------|---------|
| `data/policies.json`, `data/regions.json` | KOGL Type 1 (matches upstream Bokjiro) |
| `data/*personas*`, `data/query_pool*`, `data/*ground_truth*`, `data/gt2_*`, `data/gt3_*` | KOGL Type 1 |
| `scripts/`, `experiments/*.py` | MIT |
| `paper/*.tex` | © 2026 the authors; reuse with attribution |
