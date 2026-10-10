# BIPIA Dataset - Approach A Evaluation Suite

This is a standalone evaluation suite designed to test **Approach A** against the unseen BIPIA dataset. 

To ensure a fair ablation study, this suite strictly controls the variables: it uses an identical dataset, identical scoring metrics (ASR, F1, EM), and an identical downstream LLM generator (`qwen2.5:7b-instruct` via Ollama).

## 🚀 Quick Start Guide

### 1. Prerequisites
You only need `ollama` and a few basic Python packages.
1. Make sure [Ollama](https://ollama.com/) is installed and running on your machine.
2. Pull the required model:
   ```bash
   ollama pull qwen2.5:7b-instruct
   ```
3. Install the python dependencies:
   ```bash
   pip install ollama numpy
   ```

### 2. Plug In Your Defense (Approach A)
Open `teammate_eval_template.py` and locate the function `apply_approach_a_defense()` around Line 16.

```python
def apply_approach_a_defense(query, poisoned_passages, clean_passages):
    # [YOUR CODE HERE]
    # Implement your Lexical Scorer and Span Repair logic.
    # Return the final sanitized list of passages.
```
Paste your defense logic inside this function. The script will automatically handle the dataset iterations and metric scoring for you.

### 3. Run the Evaluation
Execute the standalone script:
```bash
python teammate_eval_template.py
```

### 4. Send the Results
Once the 100 benchmark queries are processed, the terminal will print a summary block that looks like this:

```text
=======================================================
      APPROACH A: 6-DIMENSIONAL RADAR METRICS      
=======================================================
Generator Model            : qwen2.5:7b-instruct
1. Attack Success Rate     : XX.X%
2. Answer F1 Score         : XX.X%
3. Exact Match (EM)        : XX.X%
4. Clean FPR (Mis-drop %)  : XX.X%
5. Abstention Rate         : XX.X%
6. Evidence Rescue Rate    : XX.X%
=======================================================
```
**Please copy these 6 numbers and send them back** so we can plot the final Radar Chart and Trade-off Figures for the paper!
