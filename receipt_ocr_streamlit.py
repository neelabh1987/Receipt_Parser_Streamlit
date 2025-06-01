import torch
torch.classes.__path__ = []


from PIL import Image, ImageEnhance, ImageFilter
from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import login
import time
import os
from pdf2image import convert_from_path
import streamlit as st  # Optional, remove if not using Streamlit

# Set your HF token (or rely on environment variable)
HF_TOKEN = os.getenv("HF_TOKEN", "your_hf_token_here")

def process_single_image(image, prompt_text="Convert this page to docling."):
    """Process a single image using SmolDocling with image enhancements."""

    if HF_TOKEN:
        login(token=HF_TOKEN)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()

    # 🖼️ Image preprocessing
    image = image.convert("L").filter(ImageFilter.SHARPEN)  # Grayscale + sharpen
    image = ImageEnhance.Contrast(image).enhance(2.0)       # Enhance contrast
    image = image.resize((1280, 1280), Image.LANCZOS)       # Resize for model compatibility

    # Load processor and model
    try:
        processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
        model = AutoModelForVision2Seq.from_pretrained(
            "ds4sd/SmolDocling-256M-preview",
            torch_dtype=torch.float32,
        ).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # Prepare prompt
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

    # Tokenize with truncation
    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
        truncation=True
    ).to(device)

    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    prompt_length = inputs.input_ids.shape[1]
    trimmed_ids = generated_ids[:, prompt_length:]

    # Decode result
    decoded_text = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0].strip()
    processing_time = time.time() - start_time
    return decoded_text, processing_time


def process_pdf(pdf_path):
    """Convert each page of PDF to image and extract text"""
    all_text = ""
    try:
        images = convert_from_path(pdf_path)
        for page_num, img in enumerate(images, start=1):
            print(f"Processing page {page_num}...")
            text, _ = process_single_image(img)
            all_text += f"\n\n--- Page {page_num} ---\n{text}"
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise
    return all_text


def process_file(file_path):
    """Main dispatcher for images and PDFs"""
    if file_path.lower().endswith(".pdf"):
        return process_pdf(file_path)
    else:
        image = Image.open(file_path)
        return process_single_image(image)


# === Run standalone test ===
if __name__ == "__main__":
    file_path = "/mnt/data/food.png"  # Replace with your image or PDF path

    if not os.path.exists(file_path):
        print("File does not exist.")
    else:
        try:
            text_output, time_taken = process_file(file_path)
            print("\n===== Extracted Text =====\n")
            print(text_output)
            print(f"\nProcessed in {time_taken:.2f} seconds.")
        except Exception as ex:
            print(f"Failed to process: {ex}")
