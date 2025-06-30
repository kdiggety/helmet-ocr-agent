# train_sticker_yolo_coco.py

import os
import glob
import shutil
import json
from tqdm import tqdm
from ultralytics import YOLO

def convert_coco_json_to_yolo(coco_json_path, output_dir, categories_filter=None):
    os.makedirs(output_dir, exist_ok=True)

    with open(coco_json_path) as f:
        data = json.load(f)

    category_map = {cat['id']: cat['name'] for cat in data['categories']}
    name_to_id = {v: k for k, v in category_map.items()}

    if categories_filter:
        valid_ids = [name_to_id[name] for name in categories_filter if name in name_to_id]
    else:
        valid_ids = list(category_map.keys())

    image_id_map = {}
    for img in data['images']:
        image_id_map[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }

    labels_written = 0
    for ann in tqdm(data['annotations'], desc="Converting annotations"):
        category_id = ann['category_id']
        if category_id not in valid_ids:
            continue

        image_id = ann['image_id']
        bbox = ann['bbox']  # [x_min, y_min, width, height]

        img_info = image_id_map[image_id]
        img_width = img_info['width']
        img_height = img_info['height']

        x_min, y_min, width, height = bbox
        x_center = (x_min + width / 2) / img_width
        y_center = (y_min + height / 2) / img_height
        w_norm = width / img_width
        h_norm = height / img_height

        yolo_class_id = 0  # single class 'sticker'

        label_file = os.path.splitext(img_info['file_name'])[0] + '.txt'
        label_path = os.path.join(output_dir, label_file)

        with open(label_path, 'a') as f:
            f.write(f"{yolo_class_id} {x_center} {y_center} {w_norm} {h_norm}\n")

        labels_written += 1

    print(f"✅ COCO JSON converted. Labels written: {labels_written}")

def create_dataset_yaml(yaml_path, images_dir, labels_dir, class_name='sticker'):
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(images_dir)}\n")
        f.write(f"train: {os.path.abspath(images_dir)}\n")
        f.write(f"val: {os.path.abspath(images_dir)}\n")
        f.write(f"names: ['{class_name}']\n")
    print(f"✅ YOLO dataset YAML created at {yaml_path}")

def select_best_pt_with_validation(weights_dir='runs/train', output_pt='best.pt'):
    best_pt_files = glob.glob(f"{weights_dir}/**/best.pt", recursive=True)
    best_pt_files = [f for f in best_pt_files if os.path.getsize(f) > 0]

    if not best_pt_files:
        print(f"❌ No valid best.pt found in {weights_dir}. Check training completion and dataset validity.")
        return None

    best_file = max(best_pt_files, key=os.path.getmtime)
    shutil.copy(best_file, output_pt)
    print(f"✅ Selected {best_file} -> {output_pt}")
    return best_file

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLOv8 on helmet stickers using COCO JSON annotations.")
    parser.add_argument("--coco_json", type=str, required=True, help="Path to COCO JSON annotation file")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--yolo_labels_dir", type=str, required=True, help="Output directory for YOLO labels")
    parser.add_argument("--yaml_path", type=str, default="sticker_dataset.yaml", help="Output path for YOLO dataset YAML")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--model_size", type=str, default="n", help="YOLOv8 model size (n, s, m, l, x)")
    args = parser.parse_args()

    # 1️⃣ Convert COCO JSON → YOLO TXT
    convert_coco_json_to_yolo(args.coco_json, args.yolo_labels_dir, categories_filter=['NOCSAE Recertification Sticker','Helmet Size'])

    # 2️⃣ Create dataset YAML
    create_dataset_yaml(args.yaml_path, args.images_dir, args.yolo_labels_dir)

    # 3️⃣ Train YOLOv8
    model_name = f'yolov8{args.model_size}.pt'
    model = YOLO(model_name)

    print(f"✅ Starting YOLOv8 training on {model_name}...")
    model.train(
        data=args.yaml_path,
        epochs=args.epochs,
        imgsz=640,
        project='runs/train',
        name='helmet_sticker_yolo',
        batch=16
    )

    # 4️⃣ Export model
    print("✅ Exporting YOLO model to ONNX format...")
    model.export(format='onnx')
    print("✅ Training and export complete.")

    # 5️⃣  Choose the best model
    print("✅ Choosing the best model...")
    select_best_pt(args.yaml_path)
