import os
import cv2
import re
from ultralytics import YOLO
import easyocr
from collections import Counter
import pandas as pd
import difflib

ALLOWED_SIZES = ['SMALL', 'MEDIUM', 'LARGE', 'XL']

def correct_year_text(text):
    """Extracts the first valid 4-digit year from the text, if present."""
    match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if match:
        return match.group(1)
    return None

def correct_size_text(text):
    text = text.upper().strip()
    # Use difflib to find closest match
    match = difflib.get_close_matches(text, ALLOWED_SIZES, n=1, cutoff=0.6)
    if match:
        return match[0]
    return None # fallback to raw text if no good match

def extract_year_and_size(text_list):
    year = None
    size = None
    for text in text_list:
        cleaned = text.upper().strip()
        print(f"Cleaned text: {cleaned}")

        potential_year = correct_year_text(cleaned)
        if potential_year:
            year = potential_year
            continue

        potential_size = correct_size_text(cleaned)
        if potential_size:
            size = potential_size
            continue

    return year, size

def run_pipeline_on_folder(
    folder_path="images/train",
    model_path="sticker.pt",
    output_csv="summary.csv"
):
    """
    Processes all images in folder_path:
    - YOLO to detect stickers
    - Crops bounding boxes
    - EasyOCR to extract year and size
    - Summarizes results into a CSV
    """
    reader = easyocr.Reader(['en'])

    summary = Counter()

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Skipping {filename} because the file extension is not supported")
            continue

        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        model = YOLO(model_path)

        print(f"Will begin OCR processing for {filename}")
        results = model.predict(
            img_path,
            show=False,
            save=False,
            show_labels=False,
            #batch=1,
            #stream=False,
            #verbose=True,
            conf=0.25,
            #device='cpu',
        )

        if results == None:
            print(f"Skipping {filename} because no results were found")
            continue

        for result in results:
            for idx, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)

                cropped = result.orig_img[y1:y2, x1:x2]
                cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                #text = reader.readtext(cropped, detail=0)
                text = reader.readtext(cropped_gray, detail=0)
                year, size = extract_year_and_size(text)

                if year and size:
                    summary[(year, size)] += 1
                    print(f"Detected YEAR: {year}, SIZE: {size}")
                elif year:
                    summary[(year, None)] += 1
                    print(f"Detected only YEAR: {year}")
                elif size:
                    summary[(None, size)] += 1
                    print(f"Detected only SIZE: {size}")
                else:
                    print(f"OCR Text detected but incomplete: {text}")

    # Save detailed results
    rows = [{'year': k[0], 'size': k[1], 'count': v} for k, v in summary.items()]
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Detailed OCR extraction saved to {output_csv}")

    # Create summarized count table
    summary_table = df.groupby(['year', 'size']).size().reset_index(name='count')
    summary_table = summary_table[(summary_table['year'] != "") | (summary_table['size'] != "")]
    summary_csv = "summary_counts.csv"
    summary_table.to_csv(summary_csv, index=False)
    print(f"Summary counts saved to {summary_csv}")

    return df, summary_table
