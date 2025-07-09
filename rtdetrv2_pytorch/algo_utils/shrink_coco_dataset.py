# pip install pycocotools tqdm

import json
import random
import os
import shutil
from tqdm import tqdm

def shrink_coco_dataset(
    input_json_path,
    output_json_path,
    output_image_dir=None,
    original_image_dir=None,
    num_images=100,
    seed=42
):
    # Set seed for reproducibility
    random.seed(seed)

    # Load original COCO dataset
    with open(input_json_path, 'r') as f:
        coco = json.load(f)

    # Randomly choose N images
    images = coco['images']
    selected_images = random.sample(images, min(num_images, len(images)))
    selected_image_ids = {img['id'] for img in selected_images}

    # Filter annotations to only include those that match selected images
    annotations = coco['annotations']
    selected_annotations = [ann for ann in annotations if ann['image_id'] in selected_image_ids]

    # Optionally, filter categories to only include used ones
    used_category_ids = {ann['category_id'] for ann in selected_annotations}
    selected_categories = [cat for cat in coco['categories'] if cat['id'] in used_category_ids]

    # Create new JSON
    new_coco = {
        'images': selected_images,
        'annotations': selected_annotations,
        'categories': selected_categories
    }

    # Save the new COCO file
    with open(output_json_path, 'w') as f:
        json.dump(new_coco, f, indent=2)
    
    print(f"New annotation file saved to: {output_json_path}")

    # Copy selected image files if requested
    if output_image_dir and original_image_dir:
        os.makedirs(output_image_dir, exist_ok=True)
        for img in tqdm(selected_images, desc="Copying images"):
            src_path = os.path.join(original_image_dir, img['file_name'])
            dst_path = os.path.join(output_image_dir, img['file_name'])
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: Image not found - {src_path}")

        print(f"Copied {len(selected_images)} images to: {output_image_dir}")

# Example usage
shrink_coco_dataset(
    input_json_path='/dataset/images_thermal_val/coco.json',
    output_json_path='/dataset/images_thermal_val/coco_small_128.json',
    # output_image_dir='train2017_small',               # Optional: copy selected images here
    # original_image_dir='train2017',                   # Required if copying images
    num_images=100,
    seed=1337
)

