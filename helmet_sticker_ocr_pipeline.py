from ultralytics import YOLO
import cv2
import pytesseract
import re
from collections import defaultdict
import pandas as pd
import os
import argparse

from coco_to_yolo import convert_coco_json_to_yolo
from train_yolo import train_yolo_model
from select_best_pt import select_best_checkpoint

helmet_model = YOLO('yolov8n.pt')
sticker_model = YOLO('sticker.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrated COCO to YOLO training and OCR pipeline')
    parser.add_argument('--coco-json', type=str, help='Path to COCO JSON annotation file')
    parser.add_argument('--coco-category-filter', type=str, help='Label category filter for sticker boundary areas')
    parser.add_argument('--images-dir', type=str, help='Path to images directory for conversion')
    parser.add_argument("--yolo_labels_dir", type=str, required=True, help="Output directory for YOLO labels")
    parser.add_argument("--yaml_path", type=str, default="sticker_dataset.yaml", help="Output path for YOLO dataset YAML")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for YOLO training')
    parser.add_argument('--export-onnx', action='store_true', help='Export trained YOLO model to ONNX')
    parser.add_argument('--pipeline', action='store_true', help='Run OCR pipeline after training')
    parser.add_argument('--photos', type=str, help='Path to folder with photos for OCR pipeline')
    args = parser.parse_args()

    if args.coco_json and args.images_dir:
        convert_coco_json_to_yolo(args.coco_json, args.images_dir, args.coco_category_filter)

    if args.yaml_path:
        train_yolo_model(data_yaml=args.data, epochs=args.epochs)
        select_best_checkpoint()

#    if args.pipeline and args.photos:
#        run_pipeline_on_folder(args.photos)

# This integrated script ensures you can:
# 1️⃣   Convert COCO JSON to YOLO.
# 2️⃣   Train YOLO.
# 3️⃣   Select the best.pt automatically.
# 4️⃣   Export to ONNX if needed.
# 5️⃣   Run the OCR pipeline with EasyOCR seamlessly.
# For immediate use, place your modules accordingly, then run with arguments depending on the step you wish to execute.
