import os
import json
import glob
from tqdm import tqdm
from PIL import Image

def yolo_segment_to_coco_json(images_dir, labels_dir, class_names, output_json):
    """
    Converts YOLO-format segmentation annotations into COCO-format JSON.

    Parameters:
        images_dir (str): Path to the directory containing image files (.jpg, .jpeg, .png).
        labels_dir (str): Path to the directory containing YOLO label files (.txt).
        class_names (list of str): List of class names corresponding to category IDs.
        output_json (str): Path to save the resulting COCO-format JSON file.

    This function:
        - Matches each YOLO label file with its corresponding image.
        - Converts normalized polygon coordinates into pixel values.
        - Extracts bounding boxes and area from polygons.
        - Creates and writes a COCO-format dataset including images, annotations, and categories.

    Notes:
        - Only .jpg, .jpeg, and .png image extensions are supported.
        - Each annotation must follow YOLO segmentation format: 
          <class_id> <x1> <y1> <x2> <y2> ... (normalized between 0–1).

    Output:
        - A JSON file in COCO format saved to the specified path.
    """
    
    
    image_id = 0  # Unique ID for each image
    ann_id = 0    # Unique ID for each annotation

    # Initialize COCO-format dictionary
    coco = {
        "images": [],        # List to store image metadata
        "annotations": [],   # List to store annotation data
        "categories": []     # List to store class/category info
    }

    # Populate categories with provided class names
    for i, name in enumerate(class_names):
        coco["categories"].append({
            "id": i,
            "name": name,
            "supercategory": "none"  # Not using supercategories
        })

    # Get all label files (.txt) from the labels directory
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))

    # Loop through each label file
    for label_file in tqdm(label_files, desc="Converting"):
        base_filename = os.path.splitext(os.path.basename(label_file))[0]

        # Try to find the corresponding image file (supporting jpg/jpeg/png)
        image_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            temp_path = os.path.join(images_dir, base_filename + ext)
            if os.path.exists(temp_path):
                image_path = temp_path
                break

        # If no corresponding image found, skip this label
        if image_path is None:
            continue

        # Open the image and get dimensions
        image = Image.open(image_path)
        width, height = image.size

        # Add image metadata to COCO format
        coco["images"].append({
            "file_name": os.path.basename(image_path),
            "height": height,
            "width": width,
            "id": image_id
        })

        # Read and process each line in the YOLO label file
        with open(label_file, 'r') as f:
            for line in f.readlines():
                parts = list(map(float, line.strip().split()))
                cls_id = int(parts[0])       # Class ID
                polygon_norm = parts[1:]     # Normalized polygon coordinates

                # Convert normalized coordinates (0–1) to pixel values
                polygon = []
                for i in range(0, len(polygon_norm), 2):
                    x = polygon_norm[i] * width
                    y = polygon_norm[i + 1] * height
                    polygon.append(x)
                    polygon.append(y)

                # Get bounding box from polygon
                x_coords = polygon[0::2]
                y_coords = polygon[1::2]
                x_min = min(x_coords)
                y_min = min(y_coords)
                box_width = max(x_coords) - x_min
                box_height = max(y_coords) - y_min
                area = box_width * box_height  # Area of the bounding box

                # Add annotation to COCO format
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls_id,
                    "segmentation": [polygon],  # Polygon segmentation format
                    "area": area,
                    "bbox": [x_min, y_min, box_width, box_height],  # COCO bbox format
                    "iscrowd": 0  # Assuming single object, not a crowd
                })
                ann_id += 1

        image_id += 1

    # Save final COCO-format dictionary to JSON file
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=4)

    print(f"Saved COCO-format JSON to: {output_json}")

# List of class names (category labels) for annotation
class_names = [
    "unbet", "fanduel", "draftkings", "bally", "gilariver",
    "betrivers", "bet365", "pointsbet", "betmgm", "caesars",
    "betparx", "betway", "fanatics", "casino"
]

# Convert YOLO-style segmentation annotations to COCO JSON format
yolo_segment_to_coco_json(
    images_dir="DATASET/images/val",             # Directory containing image files
    labels_dir="DATASET/labels/val",             # Directory containing YOLO label files
    class_names=class_names,                     # List of class names
    output_json="DATASET/images/val/val_segment_coco.json"  # Output COCO JSON file
)
