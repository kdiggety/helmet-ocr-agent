# Helmet Sticker OCR Agent

## Features
✅ Uses `helmet_sticker_ocr_pipeline.py` for the shared OCR logic  
✅ Jupyter + CLI remain in sync automatically
✅ Detects helmets and stickers, applies OCR, aggregates results

## Notes
✅ Leveraging MakeSense.ai to generate the COCO JSON file which contains the image regions for labels and categorizations

## 📖 Terminology

### OCR vs YOLO

| Term | Stands For | Purpose |
|------|-------------|---------|
| **OCR** | Optical Character Recognition | Reads text from images (e.g., helmet size, purchase year) |
| **YOLO** | You Only Look Once | Detects objects in images with bounding boxes (e.g., helmets, stickers) |

**OCR** converts printed or handwritten text in images into machine-readable text.  
**YOLO** is a fast object detection algorithm that identifies and localizes objects in images in a single pass.

Using **YOLO + OCR together** allows **automated helmet sticker reading workflows** without manual intervention.

## Usage

### Build:
```bash
docker build -t helmet-sticker-ocr .
```

### Run Jupyter:
```
docker run -p 8888:8888 helmet-sticker-ocr
```

### Train the Model
```bash
python -m venv yolovenv
source yolovenv/bin/activate
pip install --upgrade pip
pip install onnx onnxruntime ultralytics

# Original
python helmet_sticker_train.py --coco_json helmet_sticker_dataset.json --images_dir images/ --yolo_labels_dir labels/

# New
python -c "from coco_to_yolo import convert_coco_json_to_yolo; convert_coco_json_to_yolo('helmet_sticker_dataset.json', 'labels/train', ['NOCSAE Recertification Sticker','Helmet Size']);"
python -c "from train_yolo import train_yolo_model; train_yolo_model();"
```

### Run the Agent

# Run the Agent via Python directly

```bash
pip install pytesseract

# Original
python run_agent.py --photos 5

# New
python -c "from run_prediction_agent import run_pipeline_on_folder; run_pipeline_on_folder();
```

## Run the Agent via Python in Docker container
```
docker run --rm -v $(pwd)/photos:/app/photos -v $(pwd)/crops:/app/crops helmet-sticker-ocr python run_agent.py
```
