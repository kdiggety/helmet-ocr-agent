import os
import cv2
from ultralytics import YOLO
import easyocr
import pandas as pd

def run_pipeline_on_folder(
    folder_path="images/train",
    model_path="sticker.pt",
    crop_output_dir="crops",
    output_csv="summary.csv"
):
    """
    Processes all images in folder_path:
    - Runs YOLO model to detect sticker bounding boxes
    - Crops detected bounding boxes
    - Runs EasyOCR on each crop to extract year/size
    - Outputs a CSV summary table of counts by detected text

    Args:
        folder_path (str): Path to folder containing images
        model_path (str): Path to YOLO model weights
        crop_output_dir (str): Directory to save cropped images
        output_csv (str): CSV file to save summary table

    Returns:
        pd.DataFrame: summary dataframe
    """

    os.makedirs(crop_output_dir, exist_ok=True)

    # Load YOLO and EasyOCR
    model = YOLO(model_path)
    reader = easyocr.Reader(['en'])

    summary = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        results = model.predict(img, conf=0.3)

        for idx, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            crop = img[y1:y2, x1:x2]

            crop_filename = f"{os.path.splitext(filename)[0]}_crop_{idx}.jpg"
            crop_path = os.path.join(crop_output_dir, crop_filename)
            cv2.imwrite(crop_path, crop)

            ocr_result = reader.readtext(crop, detail=0)
            detected_text = " ".join(ocr_result).strip()

            print(f"{crop_filename} OCR: {detected_text}")

            summary.append({
                "image": filename,
                "crop": crop_filename,
                "text": detected_text
            })

    # Convert to DataFrame
    df = pd.DataFrame(summary)

    # Create summary counts table
    if not df.empty:
        summary_table = df['text'].value_counts().reset_index()
        summary_table.columns = ['text', 'count']
    else:
        summary_table = pd.DataFrame(columns=['text', 'count'])

    # Save
    summary_table.to_csv(output_csv, index=False)
    print(f"Saved summary to {output_csv}")

    return summary_table
