import json
from collections import defaultdict

# === Config ===
# input_json = "input_coco.json"
input_json='/dataset/images_rgb_val/coco.json'
output_json='/dataset/images_rgb_val/merged_coco.json'

# Old-name → new-name mapping
merge_map = {
    "person": "person",
    "bike": "vehicle",
    "car": "vehicle",
    "motor": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "other vehicle": "vehicle",
    "train": "vehicle",
    "scooter" : "vehicle"
}

# === Load original COCO JSON ===
with open(input_json, "r") as f:
    coco = json.load(f)

# Create ID ↔ name maps
id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
name_to_old_id = {cat["name"]: cat["id"] for cat in coco["categories"]}

# Build new categories from merged names
new_name_to_id = {}
new_categories = []
new_id = 1
for old_name in set(merge_map.values()):
    if old_name not in new_name_to_id:
        new_name_to_id[old_name] = new_id
        new_categories.append({
            "id": new_id,
            "name": old_name,
            "supercategory": ""
        })
        new_id += 1

# === Filter and update annotations ===
new_annotations = []
ann_id = 1

for ann in coco["annotations"]:
    old_cat_id = ann["category_id"]
    old_name = id_to_name.get(old_cat_id)

    # Skip annotation if it's not in merge_map
    if old_name not in merge_map:
        continue

    new_name = merge_map[old_name]
    new_cat_id = new_name_to_id[new_name]

    ann["category_id"] = new_cat_id
    ann["id"] = ann_id
    new_annotations.append(ann)
    ann_id += 1

# === Final COCO JSON ===
merged_coco = {
    "images": coco["images"],  # optionally filter these if needed
    "annotations": new_annotations,
    "categories": new_categories
}

# === Save output ===
with open(output_json, "w") as f:
    json.dump(merged_coco, f, indent=2)

print(f"✅ Merged and filtered COCO file saved to {output_json}")