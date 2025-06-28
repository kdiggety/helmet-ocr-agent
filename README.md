# Helmet Sticker OCR Agent

## Features
✅ Uses `ocr_pipeline.py` for single-source-of-truth logic  
✅ Jupyter + CLI remain in sync automatically  
✅ Detects helmets and stickers, applies OCR, aggregates results

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

## CLI Usage

```bash
python run_agent.py --photos 5
```

## Containerized CLI
```
docker run --rm -v $(pwd)/photos:/app/photos -v $(pwd)/crops:/app/crops helmet-sticker-ocr python run_agent.py
```
