import json
import sys
import copy

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "exported_annotations.json"
OUTPUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "migrated_" + INPUT_FILE

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

for task in data:
    for ann in task.get("annotations", []):
        new_results = []
        choices_to_add = []
        for result in ann.get("result", []):
            if result["type"] == "rectanglelabels":
                labels = result["value"].get("rectanglelabels", [])
                if "pennon_attached" in labels:
                    # Change to pennon_sail and add choices for attached
                    new_result = copy.deepcopy(result)
                    new_result["value"]["rectanglelabels"] = ["pennon_sail"]
                    new_results.append(new_result)
                    # Add corresponding choices result
                    choices_result = {
                        "id": result.get("id", None) or result.get("original_id", None) or "choice-" + str(len(new_results)),
                        "from_name": "pennon_sail_status",
                        "to_name": "label",
                        "type": "choices",
                        "value": {"choices": ["attached"]}
                    }
                    choices_to_add.append(choices_result)
                elif "pennon_detached" in labels:
                    # Change to pennon_sail and add choices for detached
                    new_result = copy.deepcopy(result)
                    new_result["value"]["rectanglelabels"] = ["pennon_sail"]
                    new_results.append(new_result)
                    # Add corresponding choices result
                    choices_result = {
                        "id": result.get("id", None) or result.get("original_id", None) or "choice-" + str(len(new_results)),
                        "from_name": "pennon_sail_status",
                        "to_name": "label",
                        "type": "choices",
                        "value": {"choices": ["detached"]}
                    }
                    choices_to_add.append(choices_result)
                elif "pennon_leech" in labels:
                    # Keep as is
                    new_results.append(result)
                else:
                    # Keep any other rectangles as is
                    new_results.append(result)
            elif result["type"] == "choices" and result.get("from_name") in ("state", "pennon_sail_status"):
                # Remove old state choices, do not keep
                continue
            else:
                # Keep all other results
                new_results.append(result)
        # Add the new choices results for pennon_sail
        new_results.extend(choices_to_add)
        ann["result"] = new_results

with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f, indent=2)

print(f"Migration complete. Output written to {OUTPUT_FILE}") 