# Helmet OCR Agent

This agent captures images of helmets with size and purchase year stickers and summarizes counts grouped by size/year.

## Features

- Capture multiple images via webcam
- Crop sticker regions from helmets
- Extract size/year via Tesseract OCR
- Output a summary table

## Run with Docker

```bash
docker build -t helmet-ocr-agent .
docker run -p 8888:8888 helmet-ocr-agent
```

## CLI Usage

```bash
python run_agent.py --photos 5
```
