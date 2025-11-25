import os
import json
import argparse
from PIL import Image
import shutil

def convert_docbank_to_coco(docbank_root, output_dir):
    """
    Convert actual DocBank dataset to COCO format
    DocBank structure should be:
    DocBank/
    ├── train/
    │   ├── images/
    │   └── annotations/ (JSON files with same name as images)
    ├── val/
    │   ├── images/
    │   └── annotations/
    └── test/
        ├── images/
        └── annotations/
    """
    
    # DocBank class mapping
    docbank_classes = {
        'abstract': 1, 'author': 2, 'caption': 3, 'equation': 4, 'figure': 5,
        'footer': 6, 'list': 7, 'paragraph': 8, 'reference': 9, 'section': 10, 'table': 11
    }
    
    def process_split(split):
        split_dir = os.path.join(docbank_root, split)
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist, skipping...")
            return
        
        images_dir = os.path.join(split_dir, 'images')
        annotations_dir = os.path.join(split_dir, 'annotations')
        
        if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
            print(f"Warning: Missing images or annotations for {split}, skipping...")
            return
        
        # Create COCO format structure
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories
        for class_name, class_id in docbank_classes.items():
            coco_data["categories"].append({
                "id": class_id,
                "name": class_name,
                "supercategory": "document"
            })
        
        image_id = 1
        annotation_id = 1
        
        # Process each image
        for img_file in os.listdir(images_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(images_dir, img_file)
            
            try:
                # Get image dimensions
                with Image.open(img_path) as img:
                    width, height = img.size
                
                # Copy image to output directory
                output_img_dir = os.path.join(output_dir, 'images')
                os.makedirs(output_img_dir, exist_ok=True)
                shutil.copy2(img_path, os.path.join(output_img_dir, img_file))
                
                # Add image info to COCO
                coco_data["images"].append({
                    "id": image_id,
                    "file_name": img_file,
                    "width": width,
                    "height": height
                })
                
                # Process corresponding annotation
                ann_file = os.path.splitext(img_file)[0] + '.json'
                ann_path = os.path.join(annotations_dir, ann_file)
                
                if os.path.exists(ann_path):
                    with open(ann_path, 'r') as f:
                        annotations = json.load(f)
                    
                    # Process each annotation in the file
                    for ann in annotations:
                        bbox = ann.get('bbox', [])
                        category = ann.get('category', '')
                        
                        if category in docbank_classes and len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            # Convert to COCO format: [x, y, width, height]
                            coco_bbox = [x1, y1, x2 - x1, y2 - y1]
                            area = (x2 - x1) * (y2 - y1)
                            
                            # Skip invalid bboxes
                            if area > 0 and coco_bbox[2] > 0 and coco_bbox[3] > 0:
                                coco_data["annotations"].append({
                                    "id": annotation_id,
                                    "image_id": image_id,
                                    "category_id": docbank_classes[category],
                                    "bbox": coco_bbox,
                                    "area": area,
                                    "iscrowd": 0,
                                    "segmentation": []  # DocBank doesn't have segmentation
                                })
                                annotation_id += 1
                
                image_id += 1
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
        
        # Save COCO annotations
        output_ann_file = os.path.join(output_dir, f'{split}.json')
        with open(output_ann_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"Converted {split}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    
    # Process all splits
    for split in ['train', 'val', 'test']:
        process_split(split)

def main():
    parser = argparse.ArgumentParser(description='Convert DocBank to COCO format')
    parser.add_argument('--docbank-root', required=True, help='Path to DocBank dataset root')
    parser.add_argument('--output-dir', required=True, help='Output directory for COCO format')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.docbank_root):
        print(f"Error: DocBank root directory {args.docbank_root} does not exist!")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    convert_docbank_to_coco(args.docbank_root, args.output_dir)

if __name__ == '__main__':
    main()