# Patient Literacy Translator Bot

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idarapatrick/Patient-Literacy-Translator-Bot/blob/main/model.ipynb)

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Technical Architecture](#technical-architecture)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Running the Project](#running-the-project)
- [Hyperparameter Experiments](#hyperparameter-experiment-results)
- [Performance Metrics](#performance-metrics)
- [Using the Interface](#using-the-gradio-interface)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

The **Patient Literacy Translator Bot** is a domain-specific Large Language Model (LLM) assistant designed for healthcare and patient advocacy. This project addresses a critical gap in healthcare communication by translating complex medical jargon into plain, accessible language that patients with no medical background can understand.

### Problem Statement

Medical professionals frequently use dense clinical terminology in discharge summaries, lab results, and clinical notes. This creates a significant barrier to health literacy, leading to:
- Poor health outcomes
- Improper medication usage
- Higher hospital readmission rates
- Patient confusion and anxiety

### Solution

This assistant acts as a **"Patient Literacy Translator"** that bridges the communication gap between healthcare providers and patients by fine-tuning a generative language model on paired complex-to-simple medical texts.

**Example Transformation:**

**Complex Medical Text:**
> MRI of the lumbar spine indicates severe spinal stenosis at the L4-L5 level with bilateral foraminal narrowing. This is causing radicular pain radiating down the posterior aspect of the lower extremities. Surgical intervention via laminectomy may be indicated if conservative management fails.

**Simplified Translation:**
> MRI of the lumbar spine shows narrowing of the spinal canal at the L4-L5 level with narrowing of the nerve openings on both sides. This is causing pain that radiates down the back of the lower legs. If conservative treatment does not work, surgery may be needed.

---

## Technical Architecture

### Base Model
- **Model:** Google Gemma-2-2b-it
- **Type:** Instruction-tuned generative language model  
- **Parameters:** 2 billion
- **Source:** [Hugging Face Model Hub](https://huggingface.co/google/gemma-2-2b-it)

### Fine-Tuning Methodology
- **Technique:** LoRA (Low-Rank Adaptation) via PEFT library
- **Approach:** Parameter-efficient fine-tuning
- **Trainable Parameters:** ~1% of total model parameters (>99% reduction)
- **Quantization:** 4-bit quantization using BitsAndBytesConfig
- **LoRA Configuration:**
  - Rank (r): 8
  - Alpha: 16
  - Dropout: 0.05
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Dataset
- **Source:** Medical Meadow WikiDoc Patient Information
- **Repository:** `medalpaca/medical_meadow_wikidoc_patient_information`
- **Training Samples:** 2,500 high-quality instruction-response pairs
- **Format:** Complex medical text → Simplified patient-friendly text
- **Preprocessing:**
  - Removal of missing values (NaNs)
  - HTML tag stripping
  - Whitespace normalization
  - Gemma-specific instruction formatting with control tokens

---

## Setup Instructions

### Prerequisites

Before starting, ensure you have:
- Google Account (for Google Colab access)
- Hugging Face Account (free registration)
- Google Drive space (~2GB for model checkpoints)

### Step 1: Create Hugging Face Account & Access Token

1. **Create Account:**
   - Visit [huggingface.co](https://huggingface.co) and sign up

2. **Generate Access Token:**
   - Navigate to Settings → Access Tokens
   - Click "New token"
   - Name: `colab-medical-translator`
   - Type: Select "Read" permissions
   - Click "Generate token"
   - **Copy and save the token securely** (you won't see it again)

3. **Accept Gemma Model License:**
   - Visit [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it)
   - Click "Agree and access repository"
   - Fill out the required form

### Step 2: Configure Google Colab

1. **Open the Notebook:**
   - Click the "Open in Colab" badge at the top of this README
   - Or manually navigate to: `https://colab.research.google.com/github/idarapatrick/Patient-Literacy-Translator-Bot/blob/main/Medical_Translator.ipynb`

2. **Add Hugging Face Token to Secrets:**
   - In Colab, locate the **Secrets** tab (key icon on the left sidebar)
   - Click "+ Add new secret"
   - **Name:** `HF_TOKEN` (must be exact)
   - **Value:** Paste your Hugging Face access token
   - Toggle "Notebook access" to ON

3. **Enable GPU Runtime:**
   - Go to **Runtime → Change runtime type**
   - **Hardware accelerator:** GPU
   - **GPU type:** T4 (recommended for free tier)
   - Click "Save"

### Step 3: Authorize Google Drive

When you run the first cell of the notebook:
1. A popup will ask for Google Drive permissions
2. Click "Connect to Google Drive"
3. Select your Google account
4. Click "Allow" to grant access

This allows the notebook to save your trained model to Google Drive for persistence.

---

## Running the Project

### Option 1: Full Pipeline Execution (Recommended)

**Best for:** First-time users who want to see the complete workflow

1. Click **Runtime → Run all** in Google Colab
2. Wait for cells to execute sequentially
3. Monitor progress in the output cells

**Estimated Runtime:**
- Environment setup: ~3-5 minutes
- Data preprocessing: ~2-3 minutes
- Hyperparameter experiments (3 runs): ~50-60 minutes
- Production training: ~60 minutes
- Evaluation & interface launch: ~5 minutes
- **Total:** ~2-2.5 hours

### Option 2: Step-by-Step Execution

**Best for:** Users who want to understand each component

#### Stage 1: Environment Setup (Cells 1-3)

**Cell 1:** Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
**Expected Output:** `Mounted at /content/drive`

**Cell 2-3:** Install dependencies and authenticate
```python
!pip install -U transformers
!pip install -q datasets peft trl bitsandbytes accelerate
```
**Duration:** ~3-5 minutes

#### Stage 2: Data Preprocessing (Cells 4-5)

**Cell 4:** Load and clean dataset
- Loads Medical Meadow dataset from Hugging Face
- Removes missing values and HTML tags
- Samples 2,500 high-quality examples

**Expected Output:**
```
Data cleaned. Rows reduced from 10000+ to 2500.
```

**Cell 5:** Format data into Gemma instruction templates
- Applies control tokens (`<start_of_turn>user`, `<start_of_turn>model`)
- Creates final training prompts

**Sample Output:**
```
<start_of_turn>user
You are a helpful medical translator...
Text to translate: [complex text]
<end_of_turn>
<start_of_turn>model
[simplified text]<end_of_turn>
```

#### Stage 3: Hyperparameter Experiments (Cell 6) - OPTIONAL

**Purpose:** Test different configurations to find optimal hyperparameters

**Experiments:**
1. **Baseline:** LR=2e-4, Batch=2, LoRA_r=8
2. **Lower LR:** LR=5e-5, Batch=2, LoRA_r=8
3. **Higher Batch:** LR=2e-4, Batch=4, LoRA_r=8
4. **Architecture:** LR=2e-4, Batch=4, LoRA_r=16

**Duration:** ~15-20 minutes per experiment (60-80 minutes total)

**Note:** You can skip this cell if you want to jump directly to production training.

#### Stage 4: Production Training (Cell 8)

**Configuration:**
- Learning Rate: 2e-4 (optimal from experiments)
- Batch Size: 4
- Gradient Accumulation: 4 steps
- Epochs: 1 full epoch
- Optimizer: PagedAdamW 8-bit

**Expected Progress:**
```
Training Progress: 100%
Step 100/100 | Loss: 1.46
Final training Total time: 60.5 minutes.
Model safely stored in Google Drive
```

**Duration:** ~60 minutes

**Best Configuration:** Experiment 3 (Learning Rate: 2e-4, Batch Size: 4, LoRA Rank: 8)
---

## Performance Metrics

### ROUGE Scores (Translation Quality)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **ROUGE-1** | 0.2436 | Measures individual word overlap with reference |
| **ROUGE-2** | 0.0731 | Measures bigram (2-word phrase) overlap |
| **ROUGE-L** | 0.1801 | Evaluates longest common subsequence (sentence structure) |

### BLEU Score
- **BLEU:** Computed during evaluation (measures n-gram precision)

### Baseline Comparison
- **Before Fine-Tuning:** Base Gemma model generated lengthy encyclopedic responses with unchanged medical jargon
- **After Fine-Tuning:** Model actively restructures sentences and replaces complex terms with plain language
- **Improvement:** >10% reduction in final loss compared to baseline

---

## Using the Gradio Interface

Once cell 11 completes, a Gradio web interface launches with a **public shareable URL**.

### Interface Layout:

```
┌─────────────────────────────────────────────────────┐
│   The Patient Literacy Translator                   │
├─────────────────────────────────────────────────────┤
│                                                      │
│  [Input Box: Paste Complex Medical Text Here]       │
│                                                      │
│  Submit                                              │
│                                                      │
│  [Output Box: Simplified Patient Translation]       │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Example Usage:

**Input:**
```
The patient presents with idiopathic peripheral neuropathy 
and requires immediate administration of analgesics for 
pain management.
```

**Output:**
```
The patient has nerve damage in their arms or legs with 
no known cause and needs pain medication right away.
```

### Sharing:
- Copy the public URL: `https://xxxxx.gradio.live`
- Share with team members or patients
- **Note:** Link expires after 72 hours or when notebook closes

---

## Project Structure

```
Patient-Literacy-Translator-Bot/
│
├── model.ipynb          # Main notebook (complete pipeline)
├── README.md                          # This comprehensive guide

```

---

## Key Dependencies

```python
transformers       # v4.36+   - Hugging Face Transformers
datasets          # v2.14+   - Hugging Face Datasets
peft              # v0.6+    - Parameter-Efficient Fine-Tuning
trl               # v0.7+    - SFTTrainer for supervised fine-tuning
bitsandbytes      # v0.41+   - 4-bit quantization
accelerate        # v0.24+   - Distributed training utilities
evaluate          # v0.4+    - Evaluation metrics
rouge_score       # v0.1+    - ROUGE implementation
gradio            # v4.0+    - Web interface framework
torch             # v2.0+    - PyTorch backend
pandas            # v2.0+    - Data manipulation
```

All dependencies are automatically installed via the notebook's pip commands.

---

## Troubleshooting

### Issue 1: Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. **Reduce batch size:**
   ```python
   per_device_train_batch_size=2  # Instead of 4
   ```

2. **Reduce max sequence length:**
   ```python
   max_seq_length=256  # Instead of 512
   ```

3. **Restart runtime:**
   - Runtime → Disconnect and delete runtime
   - Runtime → Run all

4. **Clear GPU cache:**
   ```python
   import gc
   import torch
   gc.collect()
   torch.cuda.empty_cache()
   ```

### Issue 2: Hugging Face Authentication Errors

**Symptoms:**
```
OSError: You are trying to access a gated repo.
```

**Solutions:**
1. Verify token has "Read" permissions
2. Accept Gemma license at [Hugging Face](https://huggingface.co/google/gemma-2-2b-it)
3. Ensure `HF_TOKEN` secret is correctly named (case-sensitive)
4. Check "Notebook access" toggle is ON in Secrets

### Issue 3: Slow Training Speed

**Symptoms:**
- Training takes >2 hours per experiment
- GPU utilization <50%

**Solutions:**
1. **Verify GPU is enabled:**
   ```python
   !nvidia-smi
   ```
   Should show: `Tesla T4` or similar

2. **Check runtime type:**
   - Runtime → Change runtime type → GPU

3. **Reduce dataset size for testing:**
   ```python
   df_sampled = df.sample(n=1000, random_state=42)  # Instead of 2500
   ```

### Issue 4: Gradio Interface Not Launching

**Symptoms:**
```
Error: Could not create share link
```

**Solutions:**
1. **Check all previous cells executed successfully**
2. **Verify model loaded:**
   ```python
   print(model)  # Should show PeftModel wrapper
   ```

3. **Try local launch (no share):**
   ```python
   interface.launch(share=False)
   ```

4. **Restart kernel:**
   - Runtime → Restart runtime
   - Re-run cells 10-11

### Issue 5: Google Drive Mount Fails

**Symptoms:**
```
Drive already mounted; ... or use force_remount=True
```

**Solutions:**
1. **Force remount:**
   ```python
   drive.mount('/content/drive', force_remount=True)
   ```

2. **Check permissions:**
   - Ensure you clicked "Allow" in authorization popup

3. **Use different mount point:**
   ```python
   drive.mount('/content/gdrive')
   ```
---

## Academic Context

This project was designed for academic evaluation only.
---

## License

This project uses the Google Gemma model, which is subject to **Gemma Terms of Use**.  
Please review the license at: [ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms)

---
