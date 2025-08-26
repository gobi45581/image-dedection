"""
Split a flat folder of images into train/val/test folders.
Usage:
    python split_dataset.py --input_dir all_images --output_dir dataset --train 0.7 --val 0.2 --test 0.1
"""
import argparse
from pathlib import Path
import shutil
import random

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".jfif"}

def main():
    parser = argparse.ArgumentParser(description="Split images into train/val/test")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with images")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Output root folder")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio")
    parser.addargument("--val", type=float, default=0.2, help="Val ratio")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio")
    args = parser.parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"❌ Ratios must sum to 1.0 (got {total})")

    in_dir = Path(args.input_dir)
    if not in_dir.exists():
        raise SystemExit(f"❌ Input folder not found: {in_dir.resolve()}")

    files = [p for p in in_dir.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]
    if not files:
        raise SystemExit(f"❌ No images found in {in_dir.resolve()}")

    random.shuffle(files)
    print(f"📁 Found {len(files)} images")

    out_root = Path(args.output_dir)
    for split in ["train", "val", "test"]:
        (out_root / split).mkdir(parents=True, exist_ok=True)

    n = len(files)
    n_train = int(n * args.train)
    n_val = int(n * args.val)


    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    def copy_files(subdir, file_list):
        for p in file_list:
            dst = out_root / subdir / p.name
            shutil.copy2(p, dst)

    copy_files("train", train_files)
    copy_files("val", val_files)
    copy_files("test", test_files)

    print(f"✅ Done! {n} images split -> {out_root.resolve()}")
    print(f"   Train: {len(train_files)} images ({args.train*100:.0f}%)")
    print(f"   Val:   {len(val_files)} images ({args.val*100:.0f}%)")
    print(f"   Test:  {len(test_files)} images ({args.test*100:.0f}%)")

if __name__ == "__main__":
    main()