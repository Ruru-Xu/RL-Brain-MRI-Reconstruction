import os
import json

root_path = '/home/ruru/Documents/work/BraTS/BraTs2021/preprocessing/test'
output_path = os.path.join('/home/ruru/Documents/work/journal3/segmentation/jsons', "target.json")
modalities = ["flair", "t1ce", "t1", "t2"]

data_list = []

# Traverse each case directory
for case in sorted(os.listdir(root_path)):
    case_dir = os.path.join(root_path, case)
    if os.path.isdir(case_dir):
        image_files = []
        for mod in modalities:
            img_path = os.path.join(case_dir, f"{case}_{mod}.nii.gz")
            if not os.path.isfile(img_path):
                print(f"Warning: Missing {img_path}")
            image_files.append(img_path)
        label_path = os.path.join(case_dir, f"{case}_seg.nii.gz")
        if not os.path.isfile(label_path):
            print(f"Warning: Missing {label_path}")

        entry = {
            "image": image_files,
            "label": label_path
        }
        data_list.append(entry)

# Generate JSON
dataset_json = {"testing": data_list}

# Save the JSON file
with open(output_path, "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"Saved dataset.json to {output_path}")