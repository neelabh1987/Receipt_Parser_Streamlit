import torch
import streamlit as st
import os
import time
import tempfile
from PIL import Image
from dotenv import load_dotenv
import fitz
import re

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

def clean_ocr_output(text):
    text = re.sub(r'\b(\d+)(?: \1\b){5,}', r'\1', text)  # Collapse repeated numbers
    text = re.sub(r'(?:\n?(\d+)\n?){10,}', r'\1\n', text)  # Collapse repeated lines
    text = re.sub(r'(\d+\s*){20,}', '', text)  # Remove long digit repetitions
    return text.strip()

def process_single_image(image, prompt_text):
    if HF_TOKEN:
        login(token=HF_TOKEN)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()

    processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
    model = AutoModelForVision2Seq.from_pretrained(
        "ds4sd/SmolDocling-256M-preview",
        torch_dtype=torch.float32,
    ).to(device)

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

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2
    )

    prompt_length = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, prompt_length:]

    raw_text = processor.batch_decode(trimmed_generated_ids, skip_special_tokens=True)[0].strip()
    cleaned_text = clean_ocr_output(raw_text)
    processing_time = time.time() - start_time

    return cleaned_text, processing_time

def process_pdf(pdf_file, prompt_text):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(pdf_file.read())
    temp_file.close()

    doc = fitz.open(temp_file.name)

    all_text = []
    total_time = 0

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        text, proc_time = process_single_image(image, prompt_text)
        all_text.append(f"--- Page {page_num + 1} ---\n{text}")
        total_time += proc_time

    combined_text = "\n\n".join(all_text)
    return combined_text, total_time

def main():
    st.set_page_config(page_title="OCR Text Extractor", layout="wide")
    st.title("🧾 OCR Text Extractor (Image & PDF)")

    st.write("Upload an image or PDF receipt to extract formatted OCR text using SmolDocling.")

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
        prompt_text = st.text_input(
            "Prompt for OCR (recommended default)",
            "Extract the printed receipt text line by line in plain text format. Each line should match how it appears on the bill."
        )

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
                    formatted_text, processing_time = process_single_image(image, prompt_text)
                    st.subheader("📝 Extracted Text")
                    st.text_area("OCR Result", formatted_text, height=400)
                    st.download_button("Download Text", formatted_text, file_name="ocr_output.txt")
                    st.success(f"Processing completed in {processing_time:.2f} seconds")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    elif upload_option == "PDF File" and uploaded_pdf is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                try:
                    combined_text, total_processing_time = process_pdf(uploaded_pdf, prompt_text)
                    st.subheader("📄 Extracted Text from PDF")
                    st.text_area("OCR Result", combined_text, height=400)
                    st.download_button("Download Text", combined_text, file_name="ocr_pdf_output.txt")
                    st.success(f"PDF processed in {total_processing_time:.2f} seconds")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with st.expander("ℹ️ About"):
        st.write("""
            This tool uses the [SmolDocling](https://huggingface.co/ds4sd/SmolDocling-256M-preview) model to perform OCR on uploaded images or PDFs.

            - Extracts **line-by-line text** from scanned documents.
            - Filters repetitive noise for improved readability.
            - Especially useful for **receipts, bills, invoices**.
        """)

if __name__ == "__main__":
    main()
