import argparse
import os
import sys

import cv2
import numpy as np
import torch


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def load_model(weights_path: str, stgcn_root: str):
    if stgcn_root not in sys.path:
        sys.path.insert(0, stgcn_root)

    from net.st_gcn import Model

    weights = torch.load(weights_path, map_location="cpu")
    fcn_w = weights.get("fcn.weight", None)
    if fcn_w is None:
        raise KeyError("No se encontro fcn.weight en el checkpoint.")

    num_class = int(fcn_w.shape[0])

    model = Model(
        in_channels=3,
        num_class=num_class,
        graph_args={"layout": "mediapipe_hand", "strategy": "spatial"},
        edge_importance_weighting=True,
        dropout=0.5,
    )
    model.load_state_dict(weights, strict=False)
    model.eval()
    return model


def compute_intensity(model, data_nctvm: np.ndarray):
    data = torch.from_numpy(data_nctvm).float()
    with torch.no_grad():
        _output, feature = model.extract_feature(data)

    feature = feature[0]  # (C, T_reduced, V, M)
    intensity = (feature * feature).sum(dim=0).sqrt().cpu().numpy()  # (T_reduced, V, M)
    return intensity


def normalize_intensity(intensity_tvm: np.ndarray):
    x = np.abs(intensity_tvm)
    mean_val = float(x.mean())
    if mean_val > 1e-8:
        x = x / mean_val
    return x


def draw_attention_frame(frame, coords_v2, intensity_v, edges):
    h, w = frame.shape[:2]

    skeleton = np.zeros_like(frame)
    heat = np.zeros_like(frame)

    pts = []
    for i in range(coords_v2.shape[0]):
        x = float(coords_v2[i, 0])
        y = float(coords_v2[i, 1])
        if x == 0.0 and y == 0.0:
            pts.append(None)
            continue
        px = int(np.clip(x, 0.0, 1.0) * (w - 1))
        py = int(np.clip(y, 0.0, 1.0) * (h - 1))
        pts.append((px, py))

    for i, j in edges:
        pi = pts[i] if i < len(pts) else None
        pj = pts[j] if j < len(pts) else None
        if pi is None or pj is None:
            continue
        cv2.line(skeleton, pi, pj, (255, 255, 255), 2)

    for v, p in enumerate(pts):
        if p is None:
            continue
        score = float(intensity_v[v]) if v < len(intensity_v) else 0.0
        score = max(0.0, score)
        radius = int(np.clip((score ** 0.5) * 8.0, 1.0, 42.0))
        cv2.circle(heat, p, radius, (255, 255, 255), -1)

    heat_blur = cv2.GaussianBlur(heat, (0, 0), sigmaX=7, sigmaY=7)

    overlay = frame.astype(np.float32) * 0.55
    overlay += heat_blur.astype(np.float32) * 0.75
    overlay += skeleton.astype(np.float32) * 0.35
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay


def render_attention_video(video_path, data_nctvm, intensity_tvm, out_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # input data shape: (N, C, T, V, M)
    pose = data_nctvm[0]
    t_input = pose.shape[1]
    t_feat = intensity_tvm.shape[0]

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # map frame index to pose and feature time indexes
        t_pose = min(frame_idx, t_input - 1)
        t_attn = min(int((frame_idx / max(1, t_input - 1)) * (t_feat - 1)), t_feat - 1)

        # M personas; combinamos por max para visualizacion
        xy_tvm = pose[0:2, t_pose, :, :]  # (2, V, M)
        xy_v2 = np.max(xy_tvm, axis=2).T   # (V, 2)

        attn_vm = intensity_tvm[t_attn, :, :]  # (V, M)
        attn_v = np.max(attn_vm, axis=1)       # (V,)

        out = draw_attention_frame(frame, xy_v2, attn_v, HAND_CONNECTIONS)
        writer.write(out)
        frame_idx += 1

    cap.release()
    writer.release()

    return frame_idx


def main():
    parser = argparse.ArgumentParser(description="Render de atencion ST-GCN sobre esqueleto de mano.")
    parser.add_argument("--video", required=True, help="Ruta al video original")
    parser.add_argument("--data", required=True, help="Ruta al .npy N,C,T,V,M")
    parser.add_argument("--weights", required=True, help="Ruta al checkpoint .pt")
    parser.add_argument("--stgcn_root", default="st-gcn", help="Ruta a carpeta st-gcn")
    parser.add_argument("--output", default="src/gesto_mano_attention.mp4", help="Ruta de salida MP4")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"No se encontro video: {args.video}")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"No se encontro data: {args.data}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"No se encontro checkpoint: {args.weights}")

    data = np.load(args.data)
    if data.ndim != 5:
        raise ValueError(f"Se esperaba N,C,T,V,M; recibido {data.shape}")

    model = load_model(args.weights, args.stgcn_root)
    intensity = compute_intensity(model, data)
    intensity = normalize_intensity(intensity)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    n_frames = render_attention_video(args.video, data, intensity, args.output)

    print(f"Video de atencion guardado en: {args.output}")
    print(f"Frames renderizados: {n_frames}")


if __name__ == "__main__":
    main()
