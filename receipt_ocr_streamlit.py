import torch
torch.classes.__path__ = []

# !pip install streamlit python-dotenv transformers huggingface-hub pymupdf

import streamlit as st
import os
import time
import tempfile
from PIL import Image
from dotenv import load_dotenv
import fitz  # PyMuPDF

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Check for dependencies
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from huggingface_hub import login
    transformers_available = True
except ImportError:
    transformers_available = False


def check_dependencies():
    missing = []
    if not transformers_available:
        missing.append("transformers huggingface_hub")
    return missing


def process_single_image(image, prompt_text="Convert this page to markdown."):
    """Process a single image and return Markdown-formatted text"""
    if HF_TOKEN:
        login(token=HF_TOKEN)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()

    try:
        processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
        model = AutoModelForVision2Seq.from_pretrained(
            "ds4sd/SmolDocling-256M-preview",
            torch_dtype=torch.float32,
        ).to(device)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise

    # Build prompt for Markdown
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        },
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    prompt_length = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, prompt_length:]

    markdown_text = processor.batch_decode(trimmed_generated_ids, skip_special_tokens=True)[0].strip()
    processing_time = time.time() - start_time

    return markdown_text, processing_time


def process_pdf(pdf_file, prompt_text="Convert this page to markdown."):
    """Extract Markdown-formatted text from all pages in a PDF"""
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(pdf_file.read())
    temp_file.close()

    doc = fitz.open(temp_file.name)

    all_markdown = []
    total_processing_time = 0

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        markdown_text, processing_time = process_single_image(image, prompt_text)
        all_markdown.append(f"### Page {page_num + 1}\n\n{markdown_text}")
        total_processing_time += processing_time

    combined_markdown = "\n\n".join(all_markdown)
    return combined_markdown, total_processing_time


def main():
    st.set_page_config(page_title="OCR Markdown Extractor", layout="wide")
    st.title("🧾 OCR to Markdown Extractor (Image & PDF)")

    st.write("Upload an image or PDF document to extract text as **Markdown** using SmolDocling.")

    if not HF_TOKEN:
        st.warning("⚠️ HF_TOKEN not found in .env file. Authentication may fail.")

    missing_deps = check_dependencies()
    if missing_deps:
        st.error(f"Missing dependencies: {', '.join(missing_deps)}. Please install them.")
        st.info("Install with: pip install " + " ".join(missing_deps))
        st.stop()

    with st.sidebar:
        st.header("📎 Upload Input")
        upload_option = st.radio("Choose file type:", ["Single Image", "PDF File"])
        prompt_text = st.text_input("Prompt for OCR (Markdown conversion)", "Convert this page to markdown.")
        show_raw = st.checkbox("Show raw Markdown output")

        if upload_option == "Single Image":
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        else:
            uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if upload_option == "Single Image" and uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

        if st.button("Process Image"):
            with st.spinner("Processing image..."):
                try:
                    markdown_text, processing_time = process_single_image(image, prompt_text)
                    st.subheader("📝 Extracted Markdown")
                    if show_raw:
                        st.text_area("Raw Markdown", markdown_text, height=400)
                    else:
                        st.markdown(markdown_text, unsafe_allow_html=True)
                    st.download_button("Download Markdown", markdown_text, file_name="ocr_output.md")
                    st.success(f"Processing completed in {processing_time:.2f} seconds")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    elif upload_option == "PDF File" and uploaded_pdf is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                try:
                    combined_markdown, total_processing_time = process_pdf(uploaded_pdf, prompt_text)
                    st.subheader("📄 Extracted Markdown from PDF")
                    if show_raw:
                        st.text_area("Raw Markdown", combined_markdown, height=400)
                    else:
                        st.markdown(combined_markdown, unsafe_allow_html=True)
                    st.download_button("Download Markdown", combined_markdown, file_name="ocr_pdf_output.md")
                    st.success(f"PDF processed in {total_processing_time:.2f} seconds")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with st.expander("ℹ️ About"):
        st.write("""
            This tool uses the [SmolDocling](https://huggingface.co/ds4sd/SmolDocling-256M-preview) model from Hugging Face to perform OCR and generate Markdown from scanned documents.

            - Output is **Markdown-formatted** text.
            - Useful for receipts, invoices, bills, and more.
            - Use the checkbox to toggle between rendered and raw Markdown.
        """)


if __name__ == "__main__":
    main()
