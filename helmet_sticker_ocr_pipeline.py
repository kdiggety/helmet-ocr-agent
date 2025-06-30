from ultralytics import YOLO
import cv2
import pytesseract
import re
from collections import defaultdict
import pandas as pd
import os

helmet_model = YOLO('yolov8n.pt')
sticker_model = YOLO('sticker.pt')

def detect_helmets(image):
    results = helmet_model(image)
    return [box.xyxy[0].tolist() for result in results for box in result.boxes]

def detect_stickers(helmet_crop):
    results = sticker_model(helmet_crop)
    return [box.xyxy[0].tolist() for result in results for box in result.boxes]

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

def extract_size_year(sticker_crop):
    processed = preprocess_for_ocr(sticker_crop)
    #text = pytesseract.image_to_string(processed, config='--psm 6 -c tessedit_char_whitelist="XSML0123456789"')
    text = pytesseract.image_to_string(processed, config='--psm 6 -c tessedit_char_whitelist="SMALMEDIURG0123456789"')
    match = re.search(r'(SMALL|MEDIUM|LARGE|XL|XXL)[\s\-]*(20\d{2})', text)
    return match.groups() if match else (None, None)

def process_image(image, fname_prefix="image"):
    summary = defaultdict(int)
    helmet_boxes = detect_helmets(image)
    #print(f"✅ {image} => Helmet Boxes {helmet_boxes}")
    os.makedirs("crops", exist_ok=True)
    for i, hbox in enumerate(helmet_boxes):
        x1, y1, x2, y2 = map(int, hbox)
        helmet_crop = image[y1:y2, x1:x2]
        sticker_boxes = detect_stickers(helmet_crop)
        print(f"✅ Sticker Box => {sticker_boxes}")
        for j, sbox in enumerate(sticker_boxes):
            sx1, sy1, sx2, sy2 = map(int, sbox)
            sticker_crop = helmet_crop[sy1:sy2, sx1:sx2]
            print(f"✅ Generating Crop => crops/{fname_prefix}_helmet{i}_sticker{j}.jpg")
            cv2.imwrite(f"crops/{fname_prefix}_helmet{i}_sticker{j}.jpg", sticker_crop)
            size, year = extract_size_year(sticker_crop)
            if size and year:
                summary[(size, year)] += 1
    return summary

def run_pipeline(photo_dir="images/train/"):
    summary = defaultdict(int)
    for fname in os.listdir(photo_dir):
        img = cv2.imread(os.path.join(photo_dir, fname))
        if img is None:
            print(f"❌ No images found in {photo_dir} with name {fname}")
            continue
        img_summary = process_image(img, fname_prefix=fname)
        print(f"✅ Processed image {fname}")
        print(f"✅ Image Summary {img_summary}")
        for k, v in img_summary.items():
            summary[k] += v
    df = pd.DataFrame([{'Size': s, 'Year': y, 'Count': c} for (s, y), c in summary.items()])
    return df
