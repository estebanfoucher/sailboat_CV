import argparse
import os
import json

parser = argparse.ArgumentParser(description="Convert annotations to predictions in a Label Studio tasks file.")
parser.add_argument("input_file", help="Path to the input JSON file")
args = parser.parse_args()

input_file = args.input_file
base, ext = os.path.splitext(input_file)
output_file = f"{base}_ready_to_import{ext}"

with open(input_file, "r") as f:
    tasks = json.load(f)

for task in tasks:
    # Clear predictions
    task["predictions"] = []

    # Move annotations to predictions
    for annotation in task.get("annotations", []):
        prediction = {
            "model_version": "converted_from_annotation",
            "result": annotation.get("result", []),
            "score": None,
            "created_at": annotation.get("created_at"),
            "updated_at": annotation.get("updated_at"),
            "task": task["id"],
            "project": annotation.get("project"),
            "was_cancelled": annotation.get("was_cancelled", False),
            "parent_annotation": annotation.get("id"),
        }
        task["predictions"].append(prediction)
    # Remove all annotations
    task["annotations"] = []

with open(output_file, "w") as f:
    json.dump(tasks, f, indent=2)

print(f"Done! Output written to {output_file}") 