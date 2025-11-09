import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

# -------------------------------------------------------------------
# Configuration
USERNAME = "bkqz"
GGUF_REPO_NAME = "tinyllama-quotes-generator-gguf" 
GGUF_FILE_NAME = "tinyllama-quotes-Q4_K_M.gguf"
# -------------------------------------------------------------------

# Download the GGUF model from the Hub
print(f"Downloading model {GGUF_FILE_NAME} from {USERNAME}/{GGUF_REPO_NAME}...")
model_path = hf_hub_download(
    repo_id=f"{USERNAME}/{GGUF_REPO_NAME}",
    filename=GGUF_FILE_NAME
)
print(f"Model downloaded to: {model_path}")

# Load the GGUF model using llama-cpp-python
print("Loading model with llama-cpp-python...")
try:
    llm = Llama(
        model_path=model_path,
        n_ctx=512,      # Context window size
        n_threads=os.cpu_count(), # Use all available CPU cores
        n_gpu_layers=0,  # 0 = Use CPU only
        n_batch=256
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Handle error appropriately

# The Generation Function
def generate_quote(keyword):
    """
    Generates a quote based on a user-provided keyword
    """
    if not keyword:
        return "Please provide a keyword (e.g., love, life, inspiration)."

    # Format the prompt exactly as we did during training
    prompt = f"Keyword: {keyword}\nQuote:"

    print(f"Generating quote for keyword: {keyword}")

    try:
        # Generate the text
        output = llm.create_completion(
            prompt,
            max_tokens=80,          # Max length of the generated quote
            temperature=0.7,      # Controls creativity
            top_p=0.9,
            stop=["\n", "Keyword:"], # CRITICAL: Stops generation at a newline
            echo=False              # Do not echo the prompt in the output
        )

        # Extract the generated text
        quote = output["choices"][0]["text"].strip()

        # Clean up any potential artifacts
        quote = quote.split("</s>")[0].strip()

        print(f"Generated quote: {quote}")
        return quote

    except Exception as e:
        print(f"Error during generation: {e}")
        return "Sorry, I had trouble generating a quote. Please try again."

# Create the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 💡 AI Motivational Quote Generator
        Enter a keyword (like "life", "love", "success", "hope", or "inspiration") 
        and the AI will generate a new quote for you.
        (Powered by a fine-tuned TinyLlama GGUF model)
        """
    )

    with gr.Row():
        keyword_input = gr.Textbox(
            label="Enter a Keyword", 
            placeholder="e.g., inspiration"
        )
        generate_btn = gr.Button("Generate Quote", variant="primary")

    quote_output = gr.Textbox(
        label="Generated Quote", 
        interactive=False,
        lines=5
    )

    generate_btn.click(
        fn=generate_quote,
        inputs=keyword_input,
        outputs=quote_output
    )

    gr.Examples(
        ["life", "love", "inspiration", "success", "hope", "wisdom"],
        inputs=keyword_input
    )

print("Launching Gradio app...")
demo.launch()
