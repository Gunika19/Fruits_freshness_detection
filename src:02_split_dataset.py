import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

PROCESSED   = Path("data/processed")
TRAIN_SRC   = PROCESSED / "train"
VAL_DST     = PROCESSED / "val"
CLASSES     = ["fresh", "semi_fresh", "rotten"]
VAL_RATIO   = 0.125

for cls in CLASSES:
    (VAL_DST / cls).mkdir(parents=True, exist_ok=True)

print("Splitting train → val\n")

summary = defaultdict(dict)

for cls in CLASSES:
    src_folder = TRAIN_SRC / cls
    all_images = list(src_folder.glob("*.jpg"))
    random.shuffle(all_images)

    n_val    = int(len(all_images) * VAL_RATIO)
    val_imgs = all_images[:n_val]

    for img_path in val_imgs:
        shutil.move(str(img_path), str(VAL_DST / cls / img_path.name))

    remaining = len(list(src_folder.glob("*.jpg")))
    summary[cls]["train"] = remaining
    summary[cls]["val"]   = n_val

print("Final dataset split:")
print(f"{'Class':<12} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
print("-" * 44)

test_counts = {
    "fresh":      len(list((PROCESSED / "test" / "fresh").glob("*.jpg"))),
    "semi_fresh": len(list((PROCESSED / "test" / "semi_fresh").glob("*.jpg"))),
    "rotten":     len(list((PROCESSED / "test" / "rotten").glob("*.jpg"))),
}

total_train = total_val = total_test = 0

for cls in CLASSES:
    tr  = summary[cls]["train"]
    vl  = summary[cls]["val"]
    te  = test_counts[cls]
    tot = tr + vl + te
    total_train += tr
    total_val   += vl
    total_test  += te
    print(f"{cls:<12} {tr:>8} {vl:>8} {te:>8} {tot:>8}")

grand = total_train + total_val + total_test
print("-" * 44)
print(f"{'TOTAL':<12} {total_train:>8} {total_val:>8} {total_test:>8} {grand:>8}")
print(f"\n{'Split %':<12} {total_train/grand*100:>7.1f}%"
      f" {total_val/grand*100:>7.1f}%"
      f" {total_test/grand*100:>7.1f}%")
print("\nDone! data/processed/ is ready for training.")