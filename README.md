Roll No. and Name: 2210990363 Harjot Singh  
Project Title : Evaluating-LLMs-for-UI-Element-Selection-Query-Ambiguity-and-Interface-Complexity-in-Developer-Tools   
Type : Research Paper   
Team Details : One member Harjot Singh (2210990363)   
Current Status : Submitted in Journal (ADCAIJ: Advances in Distributed Computing and Artificial Intelligence Journal) 
# Evaluating LLMs for UI Element Selection
### Query Ambiguity × Interface Complexity in Developer Tools

> **Research Paper:** *Evaluating LLMs for UI Element Selection: Query Ambiguity and Interface Complexity in Developer Tools*
> **Author:** Harjot Singh · Chitkara University, Punjab · Roll No. 2210990363
> **Supervisor:** Dr. Gurpreet Singh, Associate Professor, Dept. of CSE

---

## About This Project

This experiment evaluates how well small open-source LLMs can select the correct UI element from a list when given a natural language query — the core sub-task that any LLM-powered interface agent must perform. Two factors are systematically tested: **how ambiguous the user's query is**, and **how many elements are on the screen**.

Five models are benchmarked across 360 structured queries drawn from real GitHub and AWS interfaces, using a **3 × 2 factorial design** (3 ambiguity levels × 2 interface sizes). All 10 pairwise model comparisons are validated with McNemar's statistical significance test.

---

## Key Findings

| Model | Params | Overall Accuracy |
|---|---|---|
| TinyLlama | 1.1B | 7.22% *(near random chance)* |
| Phi-3 Mini | 3.8B | 69.72% |
| Gemma 2 | 2B | 81.67% |
| Llama 3 | 8B | 82.22% |
| **Mistral 7B** | **7B** | **90.00% ✓ Best** |

**Three critical results from the paper:**

1. **Ambiguity dominates failure** — accuracy drops by up to 36.7 percentage points going from clear to ambiguous queries, more than double the drop caused by interface complexity alone.
2. **Instruction tuning > model size** — Gemma 2 (2B) is statistically indistinguishable from Llama 3 (8B) despite having 6 billion fewer parameters (McNemar's p = 0.8918).
3. **~2B parameter threshold** — TinyLlama (1.1B) behaves randomly, indicating a minimum viable parameter count exists for structured UI tasks.

---

## Repository Structure

```
├── LLM_UI_Selection_Experiment.ipynb   ← Main experiment notebook (run this)
├── llm_ui_selection_dataset.xlsx       ← Dataset: 360 rows, 60 tasks (upload in Cell 4)
├── README.md
├── charts/                             ← Generated after running Cell 8
│   ├── 01_overall_accuracy.png
│   ├── 02_accuracy_by_ambiguity.png
│   ├── 03_accuracy_by_interface.png
│   ├── 04_accuracy_by_platform.png
│   ├── 05_heatmap.png
│   ├── 06_error_distribution.png
│   ├── 07_radar.png
│   └── 08_per_task_accuracy.png
├── results/                            ← Generated after running Cell 10
│   ├── results_phi3.csv
│   ├── results_gemma2.csv
│   ├── results_tinyllama.csv
│   ├── results_mistral.csv
│   ├── results_llama3.csv
│   └── combined_all_models.csv
└── reports/
    └── experiment_report.txt
```

---

## Dataset Overview

The dataset uses a **3 × 2 factorial structure**:

| Factor | Levels |
|---|---|
| Query Ambiguity | `clear` · `paraphrased` · `ambiguous` |
| Interface Size | `small` (10 elements) · `large` (21–24 elements) |
| Platforms | GitHub · AWS (30 tasks each) |
| Total rows | 360 (60 tasks × 6 conditions) |

**Example of the three ambiguity levels for the same task:**

| Level | Query |
|---|---|
| Clear | `"Create a new repository"` |
| Paraphrased | `"Duplicate someone else's repository to my profile"` |
| Ambiguous | `"Work on someone else's project independently"` |

---

## How to Run

The notebook is designed for **Google Colab** with a **T4 GPU**. Follow the cells in order — do not skip any cell.

### Step 0 — Open in Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook** and upload `LLM_UI_Selection_Experiment.ipynb`
3. **Change the runtime before running anything:**
   - Click **Runtime → Change runtime type**
   - Set **Hardware accelerator** to `T4 GPU`
   - Click **Save**

> ⚠️ Without a GPU, Mistral 7B and Llama 3 8B will be extremely slow or may crash. Phi-3 Mini, Gemma 2, and TinyLlama can run on CPU but will be slow.

---

### Cell 1 — Install Dependencies

```python
!pip install -q ollama pandas openpyxl matplotlib seaborn scipy groq
!curl -fsSL https://ollama.com/install.sh | sh
```

Installs all required Python packages and the Ollama model runner. Takes approximately **2 minutes**. You will see `✓ Installation complete` when done.

---

### Cell 2 — Start Ollama Service

```python
proc = subprocess.Popen(["ollama", "serve"], ...)
```

Starts the Ollama server in the background. You will see `✓ Ollama is running`. If you see `✗ Ollama not responding`, simply **re-run this cell**.

> **Note:** This cell must be re-run every time the Colab session is restarted, because Ollama does not persist between sessions.

---

### Cell 3 — Pull Models

```python
MODELS_TO_PULL = [
    "phi3:mini",    # 2.3 GB
    "gemma2:2b",    # 1.6 GB
    "tinyllama",    # 0.6 GB
    "mistral",      # 4.1 GB  ← needs GPU
    "llama3",       # 4.7 GB  ← needs GPU
]
```

Downloads the models. This is a **one-time download** — if the session is restarted you need to pull again. Takes **5–15 minutes** depending on Colab's network speed.

**To run only specific models**, comment out lines you don't need with `#`:

```python
MODELS_TO_PULL = [
    "phi3:mini",
    "gemma2:2b",
    # "tinyllama",   ← commented out, won't be pulled
    # "mistral",
    # "llama3",
]
```

You will see `✓ {model} ready` for each successful pull.

---

### Cell 4 — Upload Your Dataset

```python
uploaded = files.upload()
```

A **file picker dialog** will appear. Upload `llm_ui_selection_dataset.xlsx`. You will see a confirmation with the row count and column names.

**Required columns in the dataset file:**

| Column | Description |
|---|---|
| `task_id` | Integer 1–60 |
| `platform` | `"GitHub"` or `"AWS"` |
| `query` | The natural language query |
| `ambiguity_level` | `"clear"`, `"paraphrased"`, or `"ambiguous"` |
| `interface_size` | `"small"` or `"large"` |
| `ui_elements` | Semicolon-separated list of UI element names |
| `correct_answer` | The correct element name (exact string) |

---

### Cell 5 — Configure Models

This cell sets up the model registry and defines the prompt template. **No changes are required** unless you skipped pulling some models in Cell 3.

If you skipped pulling certain models, remove them from the `MODELS` dictionary:

```python
MODELS = {
    "phi3":      {"ollama_name": "phi3:mini",  "display": "Phi-3 Mini",  "color": "#e74c3c"},
    "gemma2":    {"ollama_name": "gemma2:2b",  "display": "Gemma 2 2B",  "color": "#3498db"},
    # "tinyllama": ...   ← remove if not pulled
}
```

The zero-shot system prompt used for all models is printed at the end of this cell for reference.

---

### Cell 6 — Run Experiment *(main step)*

```python
for mk, cfg in MODELS.items():
    for i, (_, row) in enumerate(df.iterrows()):
        ...
```

Runs every configured model on all 360 dataset rows. **This is the longest step.**

**Estimated runtimes on T4 GPU:**

| Model | Approximate Time |
|---|---|
| TinyLlama (1.1B) | ~5 minutes |
| Gemma 2 (2B) | ~10 minutes |
| Phi-3 Mini (3.8B) | ~12 minutes |
| Mistral 7B | ~20 minutes |
| Llama 3 8B | ~25 minutes |
| **All 5 models** | **~70–80 minutes total** |

Progress is printed every 30 rows and whenever a prediction is wrong. You will see the running accuracy update in real time. Each model prints a final `DONE` line with total accuracy.

> ⚠️ **Do not close the browser tab** while this cell is running. If the session times out, you will need to restart from Cell 2.

---

### Cell 7 — Compute Statistics

Calculates accuracy broken down by:
- Overall
- Ambiguity level (clear / paraphrased / ambiguous)
- Interface size (small / large)
- Platform (GitHub / AWS)
- Cross-tabulation (ambiguity × interface size, for the heatmap)

Prints a formatted summary table comparing all models side-by-side.

---

### Cell 8 — Generate Charts

Saves 8 charts to the `charts/` folder:

| File | What it shows |
|---|---|
| `01_overall_accuracy.png` | Bar chart comparing all models, with random-chance baselines |
| `02_accuracy_by_ambiguity.png` | Grouped bars — clear vs paraphrased vs ambiguous per model |
| `03_accuracy_by_interface.png` | Grouped bars — small vs large interface per model |
| `04_accuracy_by_platform.png` | Grouped bars — GitHub vs AWS per model |
| `05_heatmap.png` | Per-model heatmaps of ambiguity × interface size accuracy |
| `06_error_distribution.png` | Pie charts showing which ambiguity level caused most errors |
| `07_radar.png` | Multi-axis radar chart of model performance profiles |
| `08_per_task_accuracy.png` | Line chart of accuracy for each of the 60 tasks |

---

### Cell 9 — Statistical Significance Tests

Runs **McNemar's test** on all 10 model pairs to determine whether observed accuracy differences are statistically significant (threshold: p < 0.05).

Expected result based on the paper: all pairs are significant **except Gemma 2 (2B) vs Llama 3 (8B)** (p ≈ 0.89), which means those two models perform equivalently despite the 6B parameter gap.

---

### Cell 10 — Export Results

Saves everything and downloads a zip file:

```
experiment_results.zip
├── results/results_phi3.csv
├── results/results_gemma2.csv
├── results/combined_all_models.csv
├── charts/01_overall_accuracy.png
│   ...
└── reports/experiment_report.txt
```

The zip file downloads **automatically** to your browser's downloads folder.

---

## Troubleshooting

**`✗ Ollama not responding`**
Re-run Cell 2. The service occasionally takes longer to start on Colab.

**Model pull times out or fails**
Re-run Cell 3. Colab's internet speed varies. You can pull one model at a time by leaving only that model in `MODELS_TO_PULL`.

**Out of memory / CUDA error during Cell 6**
You likely don't have a GPU allocated. Go to **Runtime → Change runtime type → T4 GPU** and restart the session, then run from Cell 1 again.

**Dataset upload shows wrong column names**
Make sure you are uploading the correct `.xlsx` file. The required columns are listed under Cell 4 above.

**Cell 10 — `reports/` folder not found error**
Add this line before the report-writing block in Cell 10:
```python
os.makedirs("reports", exist_ok=True)
```

---

## Experimental Design Summary

The study uses a **3 × 2 factorial design** to isolate the effect of each variable:

```
Ambiguity (3 levels)    ×    Interface Size (2 levels)
  clear                         small  (10 elements,   10%  random baseline)
  paraphrased                   large  (21–24 elements, 4.5% random baseline)
  ambiguous
         ↓
   6 conditions × 60 tasks = 360 total queries
```

Each model receives a **zero-shot prompt** at temperature = 0. No examples, no chain-of-thought — just the query and the list of elements.

---

## Requirements

- Google Colab (free tier works, but T4 GPU is required for 7B/8B models)
- `llm_ui_selection_dataset.xlsx` (360 rows)
- Internet connection (for Ollama model downloads)

Python packages installed automatically in Cell 1: `ollama`, `pandas`, `openpyxl`, `matplotlib`, `seaborn`, `scipy`, `groq`

---

## Citation

If you use this experiment or dataset in your work, please cite:

```
Harjot Singh, Dr. Gurpreet Singh,
"Evaluating LLMs for UI Element Selection: Query Ambiguity and
Interface Complexity in Developer Tools",
Chitkara University, Punjab, India, 2024.
```

---

## License

This project is submitted as part of the CO-OP Project at Industry (Module-2) (22CS421) at Chitkara University. For academic use only.
