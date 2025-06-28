from ultralytics import YOLO

def train_sticker_detector():
    '''
    Trains YOLOv8 to detect helmet stickers using your labeled data.
    Ensure:
    - You have a 'sticker.yaml' dataset configuration.
    - Your images and labels are organized correctly.
    '''
    model = YOLO('yolov8n.pt')  # Using YOLOv8 nano pretrained as base
    model.train(
        data='sticker.yaml',      # Dataset configuration
        epochs=50,
        imgsz=640,
        batch=16,
        project='sticker_training_runs',
        name='yolov8_sticker',
        exist_ok=True
    )
    model.export(format='pt')    # Exports 'yolov8_sticker.pt' for your pipeline

if __name__ == "__main__":
    train_sticker_detector()