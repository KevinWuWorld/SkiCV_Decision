#!/usr/bin/env python3
"""
Convert .npz skeleton clips to MMAction2 PoseDataset pickles.
Labels: 0=clean, 1=late, 2=near-fall, 3=fall, 4=drag, -1=ignore
"""

import os, csv, pickle, glob, numpy as np
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split

def load_labels(csv_path):
    labels = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            for row in csv.reader(f):
                if not row or row[0].startswith("#") or row[0] == "clip":
                    continue
                fname, lab = row[0].strip(), int(row[1])
                labels[fname] = lab
    return labels

def npz_to_record(path, label_map):
    data = np.load(path, allow_pickle=True)
    keypoint = data["keypoint"]      # need to convert to [1,T,K,2]; training would terminate otherwise
    keypoint_score = data["keypoint_score"]   # need to convert to [1,T,K]
    total_frames = int(data["total_frames"])
    img_shape = data["img_shape"].tolist()
    fname = os.path.basename(path)

    # transpose from [T, M, V, C] to [M, T, V, C] format expected by MMAction2
    # keypoint: convert [T, 1, K, 2] to [1, T, K, 2]
    keypoint = np.transpose(keypoint, (1, 0, 2, 3)).astype(np.float32)
    # keypoint_score: convert [T, 1, K] to [1, T, K]
    keypoint_score = np.transpose(keypoint_score, (1, 0, 2)).astype(np.float32)

    label = int(label_map.get(fname, -1))
    if label < 0:
        return None  # skip unlabeled
    return {
        "frame_dir": fname.replace(".npz", ""),
        "label": label,
        "img_shape": img_shape,
        "total_frames": total_frames,
        "keypoint": keypoint,
        "keypoint_score": keypoint_score,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_dir", default="./export_skeleton_clips")
    ap.add_argument("--out_dir", default="./mmaction_skeleton")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    label_map = load_labels(os.path.join(args.export_dir, "labels.csv"))

    files = sorted(glob.glob(os.path.join(args.export_dir, "*.npz")))
    records = []
    for p in files:
        rec = npz_to_record(p, label_map)
        if rec is not None:
            records.append(rec)

    if len(records) < 2:
        raise SystemExit("Not enough labeled records. Fill labels.csv with positives & negatives first.")

    
    y = [r["label"] for r in records]
    train, val = train_test_split(records, test_size=args.val_ratio, random_state=args.seed, stratify=y)

    with open(os.path.join(args.out_dir, "train.pkl"), "wb") as f:
        pickle.dump(train, f)
    with open(os.path.join(args.out_dir, "val.pkl"), "wb") as f:
        pickle.dump(val, f)

    print(f"[ok] wrote {len(train)} train / {len(val)} val to {args.out_dir}")

if __name__ == "__main__":
    main()
