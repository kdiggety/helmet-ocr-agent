import os
import glob
import shutil
import json
from tqdm import tqdm
from ultralytics import YOLO

def train_yolo_model(data_yaml='sticker_dataset.yaml', images_dir='images', labels_dir='labels/train', epochs=50, model_size='n', categories_filter=None):

    # 1️⃣   Create dataset YAML
    create_dataset_yaml(data_yaml, images_dir, labels_dir)
    print(f"✅ Created data set YAML {data_yaml}")

    # 2️⃣   Train YOLOv8
    model_name = f'yolov8{model_size}.pt'
    model = YOLO(model_name)
    print(f"✅ Initialized YOLO model {model_name}")

    print(f"Starting YOLOv8 training on {model_name}...")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        project='runs/train',
        name='helmet_sticker_yolo',
        batch=16
    )
    print(f"✅ Completed YOLOv8 training on {model_name}...")

    # 3️⃣   Export model
    # print("✅ Exporting YOLO model to ONNX format...")
    # model.export(format='onnx')
    # print("✅ Training and export complete.")

    # 3️⃣   Choose the best model
    select_best_pt(data_yaml)

def create_dataset_yaml(data_yaml, images_dir, labels_dir, class_name='sticker'):
    with open(data_yaml, 'w') as f:
        f.write(f"path: {os.path.abspath(images_dir)}\n")
        f.write(f"train: {os.path.abspath(images_dir)}\n")
        f.write(f"val: {os.path.abspath(images_dir)}\n")
        f.write(f"names: ['{class_name}']\n")
    print(f"✅ YOLO dataset YAML created at {data_yaml}")

def select_best_pt(
    data_yaml='sticker_dataset.yaml',
    output_pt='sticker.pt',
    weights_dir='runs/train'
):
    """
    Finds the best.pt file with the highest mAP@0.5, copies it as `output_pt`, and returns the path and mAP.

    Args:
        data_yaml (str): Path to the dataset YAML for validation.
        output_pt (str): Destination filename for the selected best.pt.
        weights_dir (str): The root directoy to find candidate best.pt files.

    Returns:
        (str, float): Path to the best.pt file and its mAP@0.5, or (None, None) if not found.
    """
    pt_files = glob.glob(f"{weights_dir}/**/best.pt", recursive=True)
    if not pt_files:
        print("❌ No best.pt files found under search path.")
        return None, None

    best_map = -1
    best_pt_path = None

    for pt in pt_files:
        print(f"Evaluating {pt} ...")
        try:
            model = YOLO(pt)
            metrics = model.val(data=data_yaml, split='val', verbose=False)
            map50 = metrics.results_dict.get('metrics/mAP50(B)', 0)
            print(f"mAP50 for {pt}: {map50:.4f}")

            if map50 > best_map:
                best_map = map50
                best_pt_path = pt

        except Exception as e:
            print(f"⚠️ Error evaluating {pt}: {e}")

    if best_pt_path:
        shutil.copy(best_pt_path, output_pt)
        print(f"✅ Best model: {best_pt_path} with mAP50={best_map:.4f}")
        print(f"✅ Copied to {output_pt} for your agent pipeline.")
        return best_pt_path, best_map
    else:
        print("❌ No valid best.pt found during evaluation.")
        return None, None
