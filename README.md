# Idle penalty of local LLM servers on consumer GPUs: online appendix

This repository holds the code, raw measurements, pre-registration records and per-machine results for the paper

> Y.-C. Jian, "How Much Does Idling Cost a Local LLM Server? Measuring and Modeling Queueing Delay on Consumer GPUs," manuscript for ICAIIC 2027.

Contact: Yu-Chen Jian, Yuan Ze University, s1123708@mail.yzu.edu.tw

## Quick guide

| What you want | Where |
|---|---|
| Per-machine idle penalties (Table II) | `results/per_machine_idle_penalty.csv` |
| The 33 predictions made in advance (Table I, Fig. 2) | `results/prereg_predictions.csv` |
| Raw request and GPU logs from the RTX 4050 laptop | `LLM延遲實驗/out/` |
| Pre-registration log of the RTX 4050 experiments | `LLM延遲實驗/事前預期.md` |
| Volunteer kit, its exclusion rules, and the queue-test predictions | `朋友測試包/` |
| Every numeric sentence of the paper, recomputed from the raw data | `ICAIIC投稿/稽核/全文稽核結果.md` |

Folder names, notes and code comments are mostly in Chinese, the author's working language. This README describes each part in English.

Glossary: 閒置 idle · 排隊 queueing · 保溫 keep-warm · 校正 calibration · 預測 prediction · 結果 result · 事前 in advance · 事後 post hoc (after the results were seen) · 方法 method · 設計說明 design notes · 排除規則 exclusion rules · 稽核 audit · 朋友 volunteer.

## Setup in brief

- Server: llama.cpp release b9549 (Vulkan backend on Windows, Metal on macOS) serving `qwen2.5-coder-3b-instruct-q4_k_m.gguf` with one slot and the prompt cache disabled. On the RTX 4050 laptop: `llama-server -m <model> -c 4096 --parallel 1 -t 4 --cache-ram 0`.
- Requests: the 838-token prompt in `LLM延遲實驗/prompts/medium.txt` with a unique prefix, and 128 output tokens (`load_gen.py --max-tokens 128 --ignore-eos --unique-prompt`).
- Load: open-loop Poisson arrivals with all send times fixed in advance (`LLM延遲實驗/load_gen.py`).

## Contents

### `results/` (in English)

`per_machine_idle_penalty.csv` has one row per machine and test.

| Column | Meaning |
|---|---|
| `machine` | `author` (the RTX 4050 laptop); volunteer machines `01`–`10`, `M1` and `M1Pro` (anonymous IDs); `early-M4` and `early-HP` (first checks over Wi-Fi with 16 output tokens) |
| `type`, `device`, `test`, `power_setting` | Form factor, processor, GPU or CPU test, and the power setting for the machine that ran under two |
| `valid_rounds`, `requests` | Rounds kept under the exclusion rules / rounds run, and the number of requests |
| `back_to_back_s`, `back_to_back_cv` | Mean service time of requests sent right after the previous one completed, and its coefficient of variation |
| `penalty_3s_plus_s`, `ci95_lo_s`, `ci95_hi_s` | Extra service time after 3 s or more of idleness, with a Welch 95% confidence interval over requests |
| `share_of_back_to_back` | The penalty divided by the back-to-back service time |
| `extra_at_2s_s` … `extra_at_8s_s` | Extra service time after 2, 3, 5 and 8 s of idleness |
| `round_mean_s`, `round_ci95_lo_s`, `round_ci95_hi_s` | The penalty and its 95% interval computed from round means |
| `clear_from_s` | The first idle time whose 95% interval lies above zero and whose mean is at least 20 ms; `none` if there is none |
| `single_requests_dropped` | Requests more than three times slower than the median of their round and idle time, removed one by one under the exclusion rules |
| `source` | Where the row comes from |

`prereg_predictions.csv` has one row per prediction. The predictions for the RTX 4050, M1 and HP laptop were made before each run. The volunteer desktop runs were predicted after the kit had run but before their data were opened.

| Column | Meaning |
|---|---|
| `device`, `condition`, `run` | Machine, model or setting, and run name (for the RTX 4050, M1 and HP laptop, the name of the request log in `LLM延遲實驗/out/`) |
| `predicted_wait_s`, `range_lo_s`, `range_hi_s` | Predicted mean waiting time and its simulation range |
| `measured_wait_s` | Measured mean waiting time: waits reconstructed in completion order, with the first third of each run discarded |
| `error` | (predicted − measured) / measured |
| `inside_range` | Whether the measured value fell inside the range |
| `registered_formula_s`, `registered_formula` | The Pollaczek–Khinchine (P-K) value registered with the prediction, and which service time it used (`inf` when that service time gives ρ ≥ 1) |
| `formula_error`, `model_closer` | The error of the P-K value, and whether the measurement was closer to the model than to P-K |

### `LLM延遲實驗/` (experiments on the author's RTX 4050 laptop)

- Load generation and runs:
  - `load_gen.py`, `run_plan.py`, `run_strengthen.py`, `run_seed2.py`, `run_7b.py` and `run_7b_v2.py`.
  - Tool validation: `fake_server.py`, `run_validation.py`.
  - Idle probes: `probe_idle.py`.
  - Keep-warm: `run_power.py` and `run_keepwarm.py` (session 1), `run_keepwarm5.py` (sessions 2 and 3).
  - `run_rtx4050_supplement.py` ran the CUDA/Vulkan comparison (`run_backend_probe.py`) and keep-warm sessions 2 and 3.
- Predictions and analysis: `predict_*.py`, `analyze_*.py`, `keepwarm_model.py` (the keep-warm trade-off, in closed form and by simulation) and `check_keepwarm_runs.py`.
- `plan_*.json`: the predictions written before each run, with their simulation ranges.
- `事前預期.md`: the pre-registration log. It holds the predictions and decision rules, each entry written before the experiment it covers (in Chinese).
- `prompts/`: the request prompts, with their token counts in `token_counts.txt`.
- `llama_*.log`: llama.cpp server logs. Local paths are replaced by `<home>` or `<repo>`.
- `out/`: raw data and analysis outputs:
  - Request logs (`*.csv`). Queue runs have `t_scheduled`, `t_send`, `t_first_token` and `t_done` for each request. Idle probes (`probe_*.csv`) have the idle time before each request and its service time.
  - GPU logs (`*_gpu.csv`, no header row): time, P-state, graphics clock (MHz), memory clock (MHz), GPU utilization (%) and power (W), every 100 ms.
  - Run logs (`*_runlog.*`), console logs (`*_console.txt`), and analysis outputs (`*_summary.csv`, `*_analysis.txt`, `*.json`).

Run names in `out/`:

| Prefix | Runs |
|---|---|
| `rep_`, `rep2_` | 3B queue runs, seeds 1 and 2 |
| `var2_A_`, `var2_C_` | 3B queue runs with workloads A and C |
| `b7_`, `b7v2_` | 7B queue runs, split between GPU and CPU, and all on the GPU |
| `m1_`, `win_` | Queue runs on the Apple M1 and the HP laptop (MX250), driven over Wi-Fi |
| `my_` | Validation against the fake server |
| `probe_idle_3b*` | Idle probes on the RTX 4050 (`_heavy`: heavy keep-warm; `_nolog`: without the GPU logger) |
| `probe_base_1011`, `probe_locked_1011` | Idle probes with default and locked GPU clocks |
| `probe_m1`, `probe_m4`, `probe_wingpu`, `probe_wincpu` | Early idle checks over Wi-Fi on the M1, the M4, and the HP laptop's MX250 GPU and i7 CPU |
| `power_default_`, `power_locked_`, `kw_` | Keep-warm session 1 (endpoints, then hold times τ = 3, 6 and 12 s) |
| `kw5_`, `kw5m_` | Keep-warm sessions 2 and 3 (all five hold times in one session) |
| `backend_20261022-0041/` | CUDA versus Vulkan builds on the same laptop (requests, server logs, GPU logs) |

Earlier runs (`main_`, `main2_`, `mix_`, `long`, `var_`, `val_`, `b7s4_` and others) are kept as recorded.

### `朋友測試包/` (volunteer kit)

- `設計說明.md`: design notes for the idle kit (in Chinese).
- `排除規則.md`: the exclusion rules, fixed before the kit was sent, with dated additions (in Chinese).
- `LLM閒置測試/`: the Windows kit. `_kit/scripts/` holds `idle_test.py` (idle probes), `queue_test.py` (queue test) and `monitor.ps1` (CPU and GPU usage logger). llama.cpp and the model file are not included.
- `Mac版/`: the macOS (Metal) version.
- `排隊測試/設計說明.md`: design notes for the queue test (in Chinese). They cover prediction methods 1–5 and their decision rules, each written before the data it applies to.
  - Method 1 is the model with parameters from the calibration requests.
  - Method 2 takes the service time from the queued requests in the first third of each run.
  - Methods 3–5 let the service time follow the load.
- `分析程式/`: analysis of the volunteer results, for example `analyze_friend.py`, `summarize_all.py`, `analyze_queue.py` and `check_queue_runs.py`.
- `朋友結果/*/排隊測試_*/`: one folder for each queue test.
  - `預測_*.json`: the predictions, with the time each was saved (`made_at`). `方法N` marks method N.
  - `結果_*.json`: predictions compared with the measurements.
  - `預測與結果.md`: a write-up of the test.
  - `事後*`: post hoc analyses, not counted as predictions.
  - `重建修正_*`: the same predictions rescored after the waits were reconstructed in completion order.
  - `打開後檢查_*`: data checks made after the data were opened.

### `ICAIIC投稿/` (paper figures and audit)

- `make_paper_figs.py` (Fig. 1), `make_prereg_fig2.py` (Fig. 2) and `make_keepwarm_fig3.py` (Fig. 3).
- `稽核/全文稽核.py`: the full audit. It recomputes each numeric sentence of the paper from the raw data, and checks that every number in the paper is covered. Its output is `稽核/全文稽核結果.md`.
- `稽核/事前預測彙整.py`: collects the 33 predictions into `稽核/事前預測彙整.csv`.
- `稽核/數字清單.py`: the number inventory used by the audit.
- `稽核/check_formulas.py`: checks the closed-form mean wait against simulation.
- `稽核/check_warmup.py`: checks the warm-up cut (first third) with the MSER-5 rule.
- `稽核/recompute.py`, `稽核/check_outsample.py`: shared functions used by the scripts above. They were written for an earlier draft.

## Running the scripts

The scripts were tested with Python 3.11.7, NumPy 1.26.4, SciPy 1.17.1 and Matplotlib 3.10.9. Paths are relative to each script, so the scripts can be run from any folder, for example:

```
python ICAIIC投稿/make_keepwarm_fig3.py
python LLM延遲實驗/analyze_keepwarm5.py
```

These run from this repository alone:

- The three figure scripts.
- `稽核/check_formulas.py` and `稽核/check_warmup.py`.
- The analysis scripts in `LLM延遲實驗/` for the RTX 4050, M1 and HP-laptop runs, for example `analyze_strengthen.py`, `analyze_seed2.py`, `analyze_7b.py`, `analyze_power.py`, `analyze_keepwarm5.py`, `check_keepwarm_runs.py`, `keepwarm_model.py`, `analyze_probe.py probe_idle_3b` and `analyze_plan.py plan_m1.json`.

Some of these rewrite their outputs in `out/`. We checked that six of them come out byte for byte identical: `strengthen_summary.csv`, `idle_penalty_bins.csv`, `power_results.json`, `kw5_results.json`, `keepwarm_session_offset.json` and `keepwarm_model.json`.

Some scripts need files that are not included:

- `全文稽核.py` and `數字清單.py` read the manuscript.
- `事前預測彙整.py`, `全文稽核.py` and the scripts in `朋友測試包/分析程式/` read the volunteers' raw logs.
- `analyze_backend_probe.py` reads the original archives of the CUDA/Vulkan comparison, which also contain system information. Their logs are included unpacked in `out/backend_20261022-0041/`.

The outputs of these scripts are included. The run scripts also need llama.cpp (release b9549) and the model file, which are not included; their paths are set in each script.

## Not included

- The volunteers' raw logs. We did not ask the volunteers to publish them. Their per-machine results, as reported in the paper, are in `results/`.
- System information files (hardware inventories and `nvidia-smi -q` dumps).
- The manuscript, which will be added after review.

## AI assistance

As stated in the paper's acknowledgment, the author gave the directions and selected and revised the experimental designs. Claude Opus 5 and Claude Opus 5.5 (Anthropic) assisted by writing, debugging and running the code after the author approved each run.

## License

Code: MIT License (see `LICENSE`). Data and results: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
