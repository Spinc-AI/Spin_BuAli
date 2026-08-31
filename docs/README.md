# Documentation

Project-level documents. Each service documents itself in its own folder:
[`stt/`](../stt/README.md), [`core_llm/`](../core_llm/README.md),
[`controller/`](../controller/README.md), [`evaluation/`](../evaluation/README.md),
[`demo_app/`](../demo_app/README.md).

| File | Contents |
|---|---|
| [`Spin_BuAli_Production_Roadmap.pdf`](Spin_BuAli_Production_Roadmap.pdf) | Production readiness roadmap — 9 phases from data collection through fine-tuning, clinical safety validation, deployment and retraining, plus the infrastructure and data required (English) |
| [`Spin_BuAli_Production_Roadmap_FA.pdf`](Spin_BuAli_Production_Roadmap_FA.pdf) | The same roadmap in Persian |
| [`metrics_summary.md`](metrics_summary.md) · [`.pdf`](metrics_summary.pdf) | Evaluation metrics — the general STT set plus the clinical metrics (negation, laterality, number, unit, medical terms, critical omission, unsupported addition) and the post-edit loop (English) |
| [`metrics_summary_fa.md`](metrics_summary_fa.md) · [`.pdf`](metrics_summary_fa.pdf) | The same metrics summary in Persian |
| [`BuAli_Function_Reference.pdf`](BuAli_Function_Reference.pdf) | Function reference for the project |

> `BuAli_Function_Reference.pdf` predates the code restructure; some function
> names and locations in `controller/` and `demo_app/` have changed. For the
> current state, read each folder's own README.

## GPU server setup

For running `core_llm/` on a GPU server (CUDA install, SSH tunnel):
[`core_llm/SERVER_SETUP.md`](../core_llm/SERVER_SETUP.md).
