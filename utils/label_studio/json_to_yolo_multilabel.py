import json
import os
import sys

# Define class names and corresponding indices
class_map = {
    "pennon_sail": 0,
    "pennon_leech": 1,
}
state_map = {
    "attached": 2,
    "detached": 3,
}

def convert_annotation(annotation, image_width, image_height):
    # Build a mapping from region_id to state (choices)
    region_state = {}
    for obj in annotation:
        if obj.get("type") == "choices" and obj.get("from_name") in ("state", "pennon_sail_status"):
            region_id = obj.get("region_id")
            choices = obj["value"].get("choices", [])
            if region_id and choices:
                region_state[region_id] = choices[0]  # Only one state per region

    labels = []
    for obj in annotation:
        if obj.get("type") != "rectanglelabels":
            continue

        label = obj["value"]["rectanglelabels"][0]
        x = obj["value"]["x"] / 100
        y = obj["value"]["y"] / 100
        w = obj["value"]["width"] / 100
        h = obj["value"]["height"] / 100

        x_center = x + w / 2
        y_center = y + h / 2

        class_ids = [class_map[label]]

        # Now check for state (from choices) using region_id
        region_id = obj.get("region_id")
        if label == "pennon_sail" and region_id in region_state:
            state = region_state[region_id]
            if state in state_map:
                class_ids.append(state_map[state])

        label_line = " ".join(str(cls_id) for cls_id in class_ids)
        label_line += f" {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
        labels.append(label_line)
    return labels

def main(json_path, output_dir):
    # Load exported JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for task in data:
        image_name = os.path.basename(task["data"]["image"])
        base_name = os.path.splitext(image_name)[0]
        output_path = os.path.join(output_dir, base_name + ".txt")

        annotations = task["annotations"][0]["result"]
        width = task["data"].get("width", 1)
        height = task["data"].get("height", 1)

        labels = convert_annotation(annotations, width, height)

        with open(output_path, "w") as f:
            f.write("\n".join(labels))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python json_to_yolo_multilabel.py <export.json> [output_dir]")
        sys.exit(1)
    json_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "yolo_labels"
    main(json_path, output_dir) 