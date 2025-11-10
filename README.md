# AI Motivational Quote Generator (TinyLlama Fine-Tuning)

This project explores the process of fine-tuning the 1.1B parameter TinyLlama model to generate original motivational quotes based on a keyword.

The project evolved into two distinct phases:

1.  **V1 (Deployment):** A full end-to-end MLOps pipeline to create a CPU-based GGUF model and deploy it as a public Gradio app.

2.  **V2 (Analysis):** A deep-dive performance benchmark, added after V1, to analyze the precise impact of fine-tuning on the model's quality and speed on a GPU.

---

## 🚀 V1: Deployment Proof-of-Concept (GGUF on CPU)

This is the original goal of the project: a serverless Gradio app running a quantized GGUF model.

**Live Demo Link:**
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/bkqz/ai-quote-generator)

### Performance Warning

**Please note: This demo is a proof-of-concept for the *deployment workflow* only. The application is hosted on a free, shared CPU, which makes inference time prohibitively slow (often 5-10 minutes or more per quote).**

This real-world performance bottleneck demonstrated that this deployment method is not viable for a practical user application. This discovery was the direct motivation for the V2 Analysis.

---

## 📊 V2: Key Findings (GPU Benchmark)

Given the impractical performance of the V1 deployment, the project pivoted to a more focused analysis: **Was the fine-tuning itself a success?**

I created a second notebook (`02_gpu_performance_benchmark.ipynb`) to isolate the training and compare the **base model** vs. our **fine-tuned model** on the same T4 GPU.

The results were a clear success.

### 1. Quality (Effectiveness)

The fine-tuning successfully taught the model the new task.

* **Base Model:** Failed to adhere to the `Keyword: ... Quote:` format. The output was chaotic, including random metadata and unrelated text.
* **Fine-Tuned Model:** Achieved perfect task adherence in all tests. It consistently followed the `Keyword: ... Quote: ... - Unknown` structure and generated relevant, high-quality quotes.

### 2. Speed (Inference Performance)

This was the most significant finding. The fine-tuned model was not just better; it was **dramatically faster**.

* **Baseline Model (T4 GPU):** 51.35 seconds
* **Fine-Tuned Model (T4 GPU):** 21.06 seconds

**Project Summary:** The fine-tuning was successful. The model learned to **consistently follow the new format** in our tests, while also running **over 2.4x faster** on the same hardware.

This suggests the fine-tuning "specialized" the model. The base model was "confused" by the prompt and wasted computation on complex, varied text. The fine-tuned model has a clear, efficient generation path, which **reduced inference latency by over 50%**.

---

## 🛠️ Tech Stack

* **Model:** [TinyLlama 1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
* **Dataset:** [Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes)
* **Fine-Tuning:** PyTorch, PEFT (QLoRA), `trl` (SFTTrainer) with Early Stopping
* **Environment:** Google Colab (T4 GPU)
* **V1 Deployment:** GGUF (`Q4_K_M`) via `llama-cpp-python`
* **V2 Deployment:** LoRA Adapters pushed to Hugging Face Hub

## 📂 Repository Structure

This repository contains all artifacts for both project versions.
```
/
├── notebooks/
│   ├── 01_ai_quote_generator_gguf.ipynb         # V1: The original notebook to train and deploy the GGUF app.
│   └── 02_tinyllama_quote_finetuning.ipynb      # V2: The final notebook for quality & speed analysis.
│
├── app/
│   ├── app.py              # The Python code for the Gradio app (V1).
│   └── requirements.txt    # The app's dependencies.
│
├── .gitignore              # 
├── LICENSE                 # The MIT License,
└── README.md               # This file.
```

## 📦 Hugging Face Artifacts

All models, adapters, and the app are publicly hosted on the Hugging Face Hub.

* **V1 Gradio Demo:** [bkqz/ai-quote-generator](https://huggingface.co/spaces/bkqz/ai-quote-generator)
* **V1 GGUF Model:** [bkqz/tinyllama-quotes-generator-gguf](https://huggingface.co/bkqz/tinyllama-quotes-generator-gguf)
* **V2 LoRA Adapters:** [bkqz/tinyllama-quotes-adapters](https://huggingface.co/bkqz/tinyllama-quotes-adapters)
