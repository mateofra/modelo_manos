import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURACIÓN DE MEDIAPIPE TASKS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'hand_landmarker.task')
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

def extraer_manos_tasks(video_path):
    # Configurar opciones del Landmarker para video
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    data = []

    with HandLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convertir frame de OpenCV (BGR) a MediaPipe Image (RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Calcular timestamp en milisegundos (obligatorio para modo VIDEO)
            timestamp_ms = int((frame_idx / fps) * 1000)
            
            # Detectar landmarks
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Preparar matriz de ceros por si no se detecta nada (21 joints, 3 coordenadas)
            frame_landmarks = np.zeros((21, 3))

            if result.hand_landmarks:
                # Tomamos la primera mano detectada
                mano = result.hand_landmarks[0]
                for i, lm in enumerate(mano):
                    frame_landmarks[i] = [lm.x, lm.y, lm.z]
            
            data.append(frame_landmarks)
            frame_idx += 1

    cap.release()
    
    # --- FORMATEO PARA ST-GCN (N, C, T, V, M) ---
    # N=Batch, C=Canales(3), T=Frames, V=Joints(21), M=Persona(1)
    
    skeleton = np.array(data)            # (T, 21, 3)
    skeleton = np.expand_dims(skeleton, axis=0)  # (1, T, 21, 3)
    skeleton = skeleton.transpose(0, 3, 1, 2)    # (1, 3, T, 21)
    skeleton = np.expand_dims(skeleton, axis=-1) # (1, 3, T, 21, 1)

    return skeleton

# --- EJECUCIÓN ---
video_input = os.path.join(BASE_DIR, "gesto_mano.mp4")
archivo_salida = os.path.join(BASE_DIR, "mano_data_tasks.npy")

if os.path.exists(video_input) and os.path.exists(model_path):
    resultado = extraer_manos_tasks(video_input)
    np.save(archivo_salida, resultado)
    print(f"¡Hecho! Esqueleto guardado en {archivo_salida}")
    print(f"Dimensiones finales: {resultado.shape}") # (1, 3, Frames, 21, 1)
else:
    if not os.path.exists(model_path):
        print(f"Error: No se encuentra el archivo {model_path}. Descárgalo de Google.")
    else:
        print(f"Error: No se encontró el video {video_input}")