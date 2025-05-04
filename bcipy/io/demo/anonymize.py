import os
import json  # or use csv
import random
from pathlib import Path
import uuid


def rename_subdirs_with_map(parent_dir):
    parent = Path(parent_dir)
    subdirs = [d for d in parent.iterdir() if d.is_dir()]
    random.shuffle(subdirs)

    id_map = {}

    for idx, subdir in enumerate(subdirs):
        new_name = make_id(subdir.name)
        new_path = parent / new_name
        if subdir != new_path:
            os.rename(subdir, new_path)
            id_map[subdir.name] = new_name
            print(f"Renamed: {subdir.name} -> {new_name}")

    # Save the mapping
    map_path = parent / f"id_map.json"
    with open(map_path, "w") as f:
        json.dump(id_map, f, indent=2)

    print(f"Saved mapping to {map_path}")
    return id_map

def make_id(file_name):
    file_id = str(uuid.uuid4())[:6]
    return f"{file_id}"

if __name__ == "__main__":
    # Example usage
    target = "/Users/srikarananthoju/cambi/BciPy/bids_generated/bids_drows_78dd"
    rename_subdirs_with_map(target)
