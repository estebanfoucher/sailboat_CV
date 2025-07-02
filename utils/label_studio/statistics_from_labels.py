import json
import sys
from collections import Counter, defaultdict

# Default input file
INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "exported_annotations.json"

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

class_counter = Counter()
state_counter = Counter()
class_state_counter = defaultdict(Counter)

for task in data:
    for ann in task.get("annotations", []):
        for result in ann.get("result", []):
            if result["type"] == "rectanglelabels":
                labels = result["value"].get("rectanglelabels", [])
                for label in labels:
                    class_counter[label] += 1
                    # Optionally, associate with region_id for state lookup
                    region_id = result.get("region_id")
            elif result["type"] == "choices" and result.get("from_name") == "state":
                states = result["value"].get("choices", [])
                for state in states:
                    state_counter[state] += 1
                    # Optionally, associate with region_id for class lookup
                    region_id = result.get("region_id")
                    class_state_counter[region_id][state] += 1

print("Class statistics:")
for label, count in class_counter.items():
    print(f"  {label}: {count}")

print("\nState statistics:")
for state, count in state_counter.items():
    print(f"  {state}: {count}")

# Optionally, print class-state association if region_id is used
# print("\nClass-State statistics by region_id:")
# for region_id, states in class_state_counter.items():
#     print(f"  Region {region_id}: {dict(states)}") 