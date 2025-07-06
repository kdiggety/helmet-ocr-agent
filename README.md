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

### Create the label training JSON (https://www.makesense.ai)

### Train the Model
```bash
python -m venv yolovenv
source yolovenv/bin/activate
pip install --upgrade pip
pip install onnx onnxruntime ultralytics

echo "Original"
python helmet_sticker_train.py --coco_json helmet_sticker_dataset.json --images_dir images/ --yolo_labels_dir labels/
python -c "from ultralytics import YOLO; model = YOLO('sticker.pt'); results = model.predict('images/train/photo_3.jpeg', show=True, save=True, show_labels=False); print(f'{results}');"

echo "New"
#echo "Step 1: Generate labels..." 
#python -c "from coco_to_yolo import convert_coco_json_to_yolo; convert_coco_json_to_yolo('helmet_sticker_dataset.json', 'labels/train', ['NOCSAE Recertification Sticker','Helmet Size']);"

echo "Step 1: Normalize Image Names..." 
python -c "from normalize_names import normalize_filenames_in_sequence; normalize_filenames_in_sequence(folder='images/train');"
echo "Step 2: Train model..." 
python -c "from train_yolo import train_yolo_model; train_yolo_model();"
```

### Run the Agent

# Run the Agent via Python directly

```bash
pip install pytesseract

# Original
python run_agent.py --photos 5

echo "Only images 3, 5, and 7 have identifiable sticker labels"
python -c "import easyocr; reader = easyocr.Reader(['en'], gpu=False); results = reader.readtext('/opt/homebrew/runs/detect/predict/photo_3.jpg'); print(f'{results}')"

# New
echo "Step 1: Normalize Image Filenames"
python -c "from normalize_names import normalize_filenames_in_sequence; normalize_filenames_in_sequence(folder='images/val');"
echo "Step 2: Perform Inference"
python -c "from run_prediction_agent import run_pipeline_on_folder; run_pipeline_on_folder();
```

## Run the Agent via Python in Docker container
```
docker run --rm -v $(pwd)/photos:/app/photos -v $(pwd)/crops:/app/crops helmet-sticker-ocr python run_agent.py
```
