import os, json
from tqdm import tqdm

# Map FLIR class names to COCO-style category IDs
FLIR_CLASSES = [
    "person", "bike", "car", "motorcycle", "bus",
    "train", "truck", "traffic light", "fire hydrant",
    "street sign", "dog", "skateboard", "stroller",
    "scooter", "other vehicle"
]
category_map = {name: idx+1 for idx, name in enumerate(FLIR_CLASSES)}

def flir_to_coco(flir_json, image_dir, coco_output):
    with open(flir_json) as f:
        flir = json.load(f)

    images = []
    annotations = []
    ann_id = 1

    for img in tqdm(flir["images"]):
        images.append({
            "id": img["id"],
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        })

    for img in flir["images"]:
        img_id_index_int = int(str(img["id"]))
        current_annotation = flir["annotations"][img_id_index_int]
        # for ann in flir["annotations"].get(str(img["id"]), []):
        
        for ann in current_annotation:
            cls = ann["category"]
            if cls not in category_map: continue
            cat_id = category_map[cls]
            x, y, w, h = ann["bbox"]

            annotations.append({
                "id": ann_id,
                "image_id": img["id"],
                "category_id": cat_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": cid, "name": name} for name, cid in category_map.items()
        ]
    }

    with open(coco_output, "w") as f:
        json.dump(coco, f, indent=2)
    print("COCO JSON saved to", coco_output)

# Usage


flir_to_coco(
    flir_json="/dataset/images_thermal_train/coco.json",
    image_dir="images/",
    coco_output="/dataset/images_thermal_train/flir_coco.json"
)
