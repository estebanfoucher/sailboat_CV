import json
import os
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Convert labeled JSON to YOLO format.")
    parser.add_argument('--input', required=True, help='Path to the input JSON file')
    parser.add_argument('--output', required=True, help='Directory to save YOLO annotations')
    return parser.parse_args()

def get_status_by_id(result_list):
    """ Map object ID to its status ('attached', 'detached', etc.) """
    status_map = {}
    for item in result_list:
        if item.get('type') == 'choices':
            object_id = item.get('id')
            status = item['value']['choices'][0]
            status_map[object_id] = status
    return status_map

def convert_bbox_to_yolo(x, y, w, h):
    """ Convert absolute % bbox to YOLO format (cx, cy, w, h), normalized """
    x_center = x + w / 2
    y_center = y + h / 2
    return x_center / 100, y_center / 100, w / 100, h / 100

def process_json(json_data, output_dir):
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define label map: you can extend this as needed
    label_map = {
        "pennon_sail:attached": 0,
        "pennon_sail:detached": 1,
        "pennon_leech": 2
    }

    for task in json_data:
        image_path = task['data']['image']
        image_name = Path(image_path).name
        base_name = os.path.splitext(image_name)[0]
        annotation_path = os.path.join(output_dir, base_name + '.txt')
        
        annotations = task['annotations'][0]['result']
        status_map = get_status_by_id(annotations)

        with open(annotation_path, 'w') as f:
            for item in annotations:
                if item.get('type') != 'rectanglelabels':
                    continue
                label = item['value']['rectanglelabels'][0]
                object_id = item['id']
                width = item['value']['width']
                height = item['value']['height']
                x = item['value']['x']
                y = item['value']['y']

                orig_w = item['original_width']
                orig_h = item['original_height']

                # Convert to YOLO
                x_yolo, y_yolo, w_yolo, h_yolo = convert_bbox_to_yolo(x, y, width, height)

                # Determine class ID
                if label == "pennon_sail":
                    status = status_map.get(object_id, "attached")
                    class_key = f"{label}:{status}"
                else:
                    class_key = label

                class_id = label_map.get(class_key)
                if class_id is None:
                    print(f"Warning: Unknown class for label '{class_key}'")
                    continue

                # Write YOLO line
                f.write(f"{class_id} {x_yolo:.6f} {y_yolo:.6f} {w_yolo:.6f} {h_yolo:.6f}\n")

def main():
    args = parse_args()

    with open(args.input, 'r') as f:
        json_data = json.load(f)

    process_json(json_data, args.output)

if __name__ == "__main__":
    main()
