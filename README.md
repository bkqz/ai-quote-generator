# AI Motivational Quote Generator

A deep learning project that fine-tunes a 1.1B parameter language model (TinyLlama) to generate new, original motivational quotes based on a user-provided keyword.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/bkqz/ai-quote-generator)

This project was built by fine-tuning the model in Google Colab, converting it to the high-performance GGUF format, and deploying it in a serverless Gradio app on Hugging Face Spaces.

## 🚀 Live Demo

You can try the live application here:
**[https://huggingface.co/spaces/bkqz/ai-quote-generator](https://huggingface.co/spaces/bkqz/ai-quote-generator)**

## 🛠️ Tech Stack

* **Model:** [TinyLlama 1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
* **Dataset:** [Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes)
* **Fine-Tuning:** PyTorch, PEFT (QLoRA), `trl` (SFTTrainer)
* **Environment:** Google Colab (T4 GPU)
* **Conversion:** `llama-cpp-python` (Merged and quantized to 4-bit GGUF)
* **Deployment:** Hugging Face Spaces + Gradio

## 📖 Project Overview

This project follows a complete end-to-end MLOps workflow:

1.  **Data Preprocessing:** The `Abirate/english_quotes` dataset was formatted into a `Keyword: [TAG]\nQuote: [QUOTE]` structure to teach the model a prompt-response format.

2.  **Efficient Fine-Tuning:** The 1.1B TinyLlama model was loaded in 4-bit precision (QLoRA) on a free Google Colab T4 GPU. The `SFTTrainer` was used to efficiently fine-tune the model on the prepared quote dataset.

3.  **De-Quantization & Merging:** To prepare for GGUF conversion, the trained LoRA adapters were saved, and the base model was re-loaded in full `float16` precision. The adapters were then merged into the full-precision model.

4.  **GGUF Conversion (C++):** The merged `float16` model was converted to GGUF using the `llama.cpp` library. This involved a two-step process:
    * Converting the HF model to an intermediate `f16` GGUF.
    * Using the compiled `llama-quantize` executable to compress the `f16` file into the final `Q4_K_M` GGUF.

5.  **Deployment:** The final ~670MB GGUF model was uploaded to **[Hugging Face Hub](https://huggingface.co/bkqz/tinyllama-quotes-generator-gguf)**. A separate **Hugging Face Space** hosts the Gradio app, which downloads the GGUF file and runs it on a free CPU using `llama-cpp-python` for fast inference.

## Files in this Repository

* `ai_quote_generator.ipynb`: The complete Google Colab notebook used for training, merging, and GGUF conversion.
* `app.py`: The Python script for the Gradio application.
* `requirements.txt`: The dependencies required by the Gradio app.
