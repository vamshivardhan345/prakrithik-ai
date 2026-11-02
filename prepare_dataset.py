import os
import zipfile
import shutil
import random
import json
from PIL import Image
from collections import defaultdict

ZIP_PATH = r"C:\Users\suvar\Downloads\archive (16).zip"
OUTPUT_DIR = r"dataset"
RESULTS_DIR = r"results"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

train_dir = os.path.join(OUTPUT_DIR, "train")
val_dir = os.path.join(OUTPUT_DIR, "validation")
test_dir = os.path.join(OUTPUT_DIR, "test")

for d in [train_dir, val_dir, test_dir]:
    os.makedirs(d, exist_ok=True)

print("Reading zip file structure...")
z = zipfile.ZipFile(ZIP_PATH, 'r')

# Collect all image files by class
class_files = defaultdict(list)
for member in z.infolist():
    if member.is_dir():
        continue
    name = member.filename
    if name.lower().endswith(('.jpg', '.jpeg', '.png')):
        parts = name.split('/')
        if len(parts) >= 3:
            cls_name = parts[2]
            class_files[cls_name].append(member)

classes = sorted(list(class_files.keys()))
print(f"Found {len(classes)} classes in dataset archive.")

# Save class list
with open(os.path.join("model", "classes.json"), "w") as f:
    json.dump(classes, f, indent=4)
print("Saved model/classes.json")

report = {
    "classes": {},
    "total_valid": 0,
    "total_corrupted": 0,
    "total_train": 0,
    "total_val": 0,
    "total_test": 0
}

random.seed(42)

for idx, cls in enumerate(classes, 1):
    cls_members = class_files[cls]
    random.shuffle(cls_members)
    
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
    
    valid_members = []
    corrupted_count = 0
    
    # Validate each image
    for member in cls_members:
        try:
            with z.open(member) as f:
                img = Image.open(f)
                img.verify()  # Check for corruption
            valid_members.append(member)
        except Exception:
            corrupted_count += 1

    total_valid = len(valid_members)
    
    # 70% train, 15% validation, 15% test
    n_train = int(total_valid * 0.70)
    n_val = int(total_valid * 0.15)
    n_test = total_valid - n_train - n_val
    
    train_members = valid_members[:n_train]
    val_members = valid_members[n_train:n_train + n_val]
    test_members = valid_members[n_train + n_val:]
    
    # Extract to target directories
    for member in train_members:
        fname = os.path.basename(member.filename)
        dest = os.path.join(train_dir, cls, fname)
        with z.open(member) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
            
    for member in val_members:
        fname = os.path.basename(member.filename)
        dest = os.path.join(val_dir, cls, fname)
        with z.open(member) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
            
    for member in test_members:
        fname = os.path.basename(member.filename)
        dest = os.path.join(test_dir, cls, fname)
        with z.open(member) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
            
    report["classes"][cls] = {
        "total": len(cls_members),
        "valid": total_valid,
        "corrupted": corrupted_count,
        "train": len(train_members),
        "validation": len(val_members),
        "test": len(test_members)
    }
    
    report["total_valid"] += total_valid
    report["total_corrupted"] += corrupted_count
    report["total_train"] += len(train_members)
    report["total_val"] += len(val_members)
    report["total_test"] += len(test_members)
    
    if idx % 5 == 0 or idx == len(classes):
        print(f"[{idx}/{len(classes)}] Processed {cls}: {total_valid} valid ({len(train_members)} train, {len(val_members)} val, {len(test_members)} test)")

# Save detailed report
report_path_json = os.path.join(RESULTS_DIR, "dataset_distribution_report.json")
with open(report_path_json, "w") as f:
    json.dump(report, f, indent=4)

report_path_txt = os.path.join(RESULTS_DIR, "dataset_distribution_report.txt")
with open(report_path_txt, "w") as f:
    f.write("PLANTVILLAGE DATASET STRATIFIED SPLIT REPORT\n")
    f.write("=" * 75 + "\n")
    f.write(f"{'Class Name':<45} | {'Total':<6} | {'Train':<6} | {'Val':<6} | {'Test':<6}\n")
    f.write("-" * 75 + "\n")
    for cls, data in sorted(report["classes"].items()):
        f.write(f"{cls:<45} | {data['valid']:<6} | {data['train']:<6} | {data['validation']:<6} | {data['test']:<6}\n")
    f.write("=" * 75 + "\n")
    f.write(f"Total Valid Images: {report['total_valid']}\n")
    f.write(f"Total Corrupted Removed: {report['total_corrupted']}\n")
    f.write(f"Training Set (70%): {report['total_train']}\n")
    f.write(f"Validation Set (15%): {report['total_val']}\n")
    f.write(f"Testing Set (15%): {report['total_test']}\n")

print(f"\nCompleted! Total valid: {report['total_valid']}, Train: {report['total_train']}, Val: {report['total_val']}, Test: {report['total_test']}")
