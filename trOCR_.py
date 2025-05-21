from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2
from device import device
import re
import os

cwd = os.path.dirname(os.path.abspath(__file__))

def ocr_trocr(image, processor, model):
    img_rgb      = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = processor(img_rgb, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if processor is processor_base:
        text = re.sub(r'[^0-9]', '', text)
    return text.strip()

trocr_mssv_path = os.path.join(cwd, r"trocr-mssv")
trocr_score_path = os.path.join(cwd,r"trocr-score")

# TrOCR cho MSSV
processor_base = TrOCRProcessor.from_pretrained(trocr_mssv_path)
trocr_base = VisionEncoderDecoderModel.from_pretrained(trocr_mssv_path)

# TrOCR cho Score
processor_score = TrOCRProcessor.from_pretrained(trocr_score_path)
trocr_score = VisionEncoderDecoderModel.from_pretrained(trocr_score_path)
