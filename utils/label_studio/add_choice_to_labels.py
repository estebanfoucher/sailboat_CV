import json

with open("exported_annotations.json") as f:
    data = json.load(f)

for task in data:
    for ann in task.get("annotations", []):
        for result in ann["result"]:
            if result["type"] == "rectanglelabels":
                # Add placeholder for state classification
                new_result = {
                    "from_name": "state",
                    "to_name": "image",
                    "type": "choices",
                    "value": {"choices": ["attached"]},
                    "original_width": result["original_width"],
                    "original_height": result["original_height"],
                    "image_rotation": result.get("image_rotation", 0),
                    "id": result.get("id", ""),
                    "region_id": result.get("region_id")
                }
                ann["result"].append(new_result)

# Save new task set
with open("updated_tasks.json", "w") as f:
    json.dump(data, f, indent=2)
