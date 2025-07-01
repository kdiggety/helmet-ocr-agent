import os
import json
from tqdm import tqdm

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
