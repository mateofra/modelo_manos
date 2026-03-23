import argparse
import os
import pickle

import numpy as np


def adapt_joint_count(data: np.ndarray, target_joints: int) -> np.ndarray:
    """Adapt V dimension from input joints to target_joints by pad/truncate."""
    n, c, t, v, m = data.shape
    if v == target_joints:
        return data

    out = np.zeros((n, c, t, target_joints, m), dtype=data.dtype)
    copy_v = min(v, target_joints)
    out[:, :, :, :copy_v, :] = data[:, :, :, :copy_v, :]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare custom N,C,T,V,M numpy data for ST-GCN feeder format."
    )
    parser.add_argument("--input", required=True, help="Path to input .npy (N,C,T,V,M)")
    parser.add_argument(
        "--out_dir",
        default="st-gcn/data/custom-hand",
        help="Output directory where train/val files are written",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=0,
        help="Label index for this sample (default: 0)",
    )
    parser.add_argument(
        "--sample_name",
        default="sample_000",
        help="Sample name to store in label pkl",
    )
    parser.add_argument(
        "--target_joints",
        type=int,
        default=21,
        choices=[18, 21, 24, 25],
        help="Target joint count to match graph layout (21 for mediapipe_hand, 25 for ntu-rgb+d)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    data = np.load(args.input)
    if data.ndim != 5:
        raise ValueError(f"Expected 5D N,C,T,V,M array, got shape {data.shape}")

    data = adapt_joint_count(data, args.target_joints)

    os.makedirs(args.out_dir, exist_ok=True)

    sample_name = [args.sample_name]
    sample_label = [args.label]

    for part in ("train", "val"):
        np.save(os.path.join(args.out_dir, f"{part}_data.npy"), data)
        with open(os.path.join(args.out_dir, f"{part}_label.pkl"), "wb") as f:
            pickle.dump((sample_name, sample_label), f)

    print("Prepared ST-GCN custom dataset:")
    print(f"  input: {args.input}")
    print(f"  out_dir: {args.out_dir}")
    print(f"  shape: {data.shape}")
    print(f"  label: {args.label}")


if __name__ == "__main__":
    main()
