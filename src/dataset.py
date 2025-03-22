import os
import random
import shutil
import kagglehub

path = kagglehub.dataset_download("muhammadsaoodsarwar/drone-vs-bird") + "/dataset"

OUTPUT_DIR = "data"
splits = ["train", "val", "test"]
classes = ["bird", "drone"]


def create_dataset(
    src_dir: str, class_name: str, train_size: int, val_size: int
) -> None:

    if train_size + val_size >= 1:
        raise ValueError("train_size + val_size should be less than 1")

    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    files = [
        f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    random.shuffle(files)
    total = len(files)
    train_end = int(train_size * total)
    val_end = train_end + int(val_size * total)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    for f in train_files:
        shutil.copy(
            os.path.join(src_dir, f), os.path.join(OUTPUT_DIR, "train", class_name, f)
        )
    for f in val_files:
        shutil.copy(
            os.path.join(src_dir, f), os.path.join(OUTPUT_DIR, "val", class_name, f)
        )
    for f in test_files:
        shutil.copy(
            os.path.join(src_dir, f), os.path.join(OUTPUT_DIR, "test", class_name, f)
        )


create_dataset(path + "/bird", "bird", train_size=0.7, val_size=0.15)
create_dataset(path + "/drone", "drone", train_size=0.7, val_size=0.15)
