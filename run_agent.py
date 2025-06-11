import cv2
import pytesseract
from PIL import Image
import numpy as np
import re
import os
from collections import defaultdict
import argparse
import pandas as pd

def capture_images(num_photos, output_dir="photos"):
    os.makedirs(output_dir, exist_ok=True)
    cam = cv2.VideoCapture(0)
    for i in range(num_photos):
        ret, frame = cam.read()
        if ret:
            cv2.imwrite(f"{output_dir}/photo_{i}.jpg", frame)
    cam.release()

def extract_sticker_regions(image):
    h, w, _ = image.shape
    crops = [
        image[int(h*0.7):h, int(w*0.05):int(w*0.25)],
        image[int(h*0.7):h, int(w*0.35):int(w*0.55)]
    ]
    return crops

def extract_size_year(crop_img):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    pil_img = Image.fromarray(thresh)
    text = pytesseract.image_to_string(pil_img)
    match = re.search(r'(XS|S|M|L|XL|XXL)[\s\-]*(20\d{2})', text)
    return match.groups() if match else (None, None)

def run_pipeline(photo_dir, num_photos):
    summary = defaultdict(int)
    for i in range(num_photos):
        path = f"{photo_dir}/photo_{i}.jpg"
        img = cv2.imread(path)
        if img is None:
            continue
        crops = extract_sticker_regions(img)
        for crop in crops:
            size, year = extract_size_year(crop)
            if size and year:
                summary[(size, year)] += 1
    df = pd.DataFrame([{'Size': s, 'Year': y, 'Count': c} for (s, y), c in summary.items()])
    return df.sort_values(['Year', 'Size'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos", type=int, default=5, help="Number of photos to capture")
    args = parser.parse_args()

    capture_images(args.photos)
    df = run_pipeline("photos", args.photos)
    print(df.to_string(index=False))
