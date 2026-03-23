import os
import argparse

import cv2
import imageio.v2 as imageio
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
INPUT_VIDEO = os.path.join(BASE_DIR, "gesto_mano.mp4")
GIF_MAX_FRAMES = 48
GIF_STRIDE = 2

# MediaPipe Hands canonical connections (21 joints).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_hand_landmarks(frame, detection_result):
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        points = []
        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))

        for i, j in HAND_CONNECTIONS:
            if i < len(points) and j < len(points):
                cv2.line(annotated, points[i], points[j], (0, 255, 0), 2)

        for x, y in points:
            cv2.circle(annotated, (x, y), 3, (0, 0, 255), -1)

        handedness_label = "Unknown"
        handedness_score = 0.0
        if hasattr(detection_result, "handedness") and hand_idx < len(detection_result.handedness):
            categories = detection_result.handedness[hand_idx]
            if categories and len(categories) > 0:
                handedness_label = categories[0].category_name
                handedness_score = float(categories[0].score)

        if points:
            tx, ty = points[0]
            ty = max(20, ty - 10)
            text = f"{handedness_label} {handedness_score:.2f}"
            cv2.putText(
                annotated,
                text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

    return annotated


def build_default_outputs(input_video):
    base, _ext = os.path.splitext(input_video)
    output_video = f"{base}_skeleton.mp4"
    output_video_black = f"{base}_skeleton_black.mp4"
    output_gif = f"{base}_preview.gif"
    return output_video, output_video_black, output_gif


def main():
    parser = argparse.ArgumentParser(description="Dibuja esqueleto de manos en video con MediaPipe.")
    parser.add_argument("--input", default=INPUT_VIDEO, help="Ruta del video de entrada")
    parser.add_argument("--model", default=MODEL_PATH, help="Ruta del modelo hand_landmarker.task")
    parser.add_argument("--num_hands", type=int, default=2, help="Cantidad maxima de manos a detectar")
    parser.add_argument("--output", default=None, help="Ruta del mp4 de salida con overlay")
    parser.add_argument("--output_black", default=None, help="Ruta del mp4 de salida con fondo negro")
    parser.add_argument("--output_gif", default=None, help="Ruta del gif de vista previa")
    args = parser.parse_args()

    output_video, output_video_black, output_gif = build_default_outputs(args.input)
    if args.output:
        output_video = args.output
    if args.output_black:
        output_video_black = args.output_black
    if args.output_gif:
        output_gif = args.output_gif

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"No se encuentra el modelo: {args.model}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"No se encuentra el video: {args.input}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Codec compatible con MP4 en la mayoria de entornos.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    writer_black = cv2.VideoWriter(output_video_black, fourcc, fps, (width, height))
    gif_frames = []

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=args.model),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.num_hands,
        min_hand_detection_confidence=0.5,
    )

    frame_idx = 0
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / fps) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            annotated = draw_hand_landmarks(frame, result)
            black_frame = draw_hand_landmarks(
                cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) * 0,
                result,
            )
            writer.write(annotated)
            writer_black.write(black_frame)

            if frame_idx % GIF_STRIDE == 0 and len(gif_frames) < GIF_MAX_FRAMES:
                gif_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            frame_idx += 1

    cap.release()
    writer.release()
    writer_black.release()

    if gif_frames:
        imageio.mimsave(output_gif, gif_frames, duration=1.0 / max(1.0, fps / GIF_STRIDE))

    print(f"Video con esqueleto guardado en: {output_video}")
    print(f"Video con fondo negro guardado en: {output_video_black}")
    print(f"GIF de vista previa guardado en: {output_gif}")
    print(f"Frames procesados: {frame_idx}")


if __name__ == "__main__":
    main()
