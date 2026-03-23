# Flujo realizado hasta ahora

## 1) Instalar dependencias

```powershell
uv add -r .\st-gcn\requirements.txt
uv sync
```

## 2) Descargar video y convertir a MP4

Se descargó el video de Wikimedia y se guardó en:

- `src/gesto_mano.mp4`

## 3) Extraer landmarks de mano con MediaPipe

Script usado:

- `src/generate.py`

Comando:

```powershell
uv run python .\src\generate.py
```

Salida generada:

- `src/mano_data_tasks.npy`
- Forma: `(1, 3, T, 21, 1)`

## 4) Preparar dataset para ST-GCN

Script usado:

- `src/prepare_custom_stgcn_data.py`

Comando (21 joints reales):

```powershell
uv run python .\src\prepare_custom_stgcn_data.py --input .\src\mano_data_tasks.npy --out_dir .\st-gcn\data\custom-hand-21 --label 0 --sample_name gesto_mano
```

Archivos generados:

- `st-gcn/data/custom-hand-21/train_data.npy`
- `st-gcn/data/custom-hand-21/train_label.pkl`
- `st-gcn/data/custom-hand-21/val_data.npy`
- `st-gcn/data/custom-hand-21/val_label.pkl`

## 5) Añadir layout de grafo para MediaPipe Hands (21 joints)

Archivo modificado:

- `st-gcn/net/utils/graph.py`

Layout agregado:

- `mediapipe_hand` (21 nodos)

## 6) YAML listo para correr

Archivo creado:

- `st-gcn/config/st_gcn/custom-hand-21/train.yaml`

Este YAML usa:

- `graph_args.layout: mediapipe_hand`
- `data/custom-hand-21` como fuente de train/val
- CPU por defecto (`use_gpu: False`)

## 7) Ejecutar entrenamiento (prueba rápida validada)

Desde la carpeta `st-gcn`:

```powershell
cd .\st-gcn
$env:PYTHONPATH = (Resolve-Path .\torchlight).Path
uv run python main.py recognition -c config/st_gcn/custom-hand-21/train.yaml --num_epoch 1
```

Validación:

- Entrenamiento y evaluación completados correctamente
- Checkpoint guardado en `st-gcn/work_dir/recognition/custom-hand-21/ST_GCN/epoch1_model.pt`

## 8) Visualización del esqueleto extraído (ambas manos)

Script creado:

- `src/draw_skeleton_mediapipe.py`

Qué hace:

- Detecta y dibuja manos por frame (por defecto `num_hands=2`, configurable por CLI)
- Dibuja landmarks y conexiones del esqueleto de mano
- Muestra etiqueta `Left/Right` y score de confianza por mano
- Exporta 3 salidas visuales

Parámetros CLI disponibles:

- `--input`: ruta del video de entrada
- `--num_hands`: máximo de manos a detectar por frame
- `--model`: ruta del archivo `hand_landmarker.task`
- `--output`, `--output_black`, `--output_gif`: rutas de salida personalizadas

Comando de ejecución desde raíz del proyecto:

```powershell
uv run python .\src\draw_skeleton_mediapipe.py
```

Comando de ejecución desde `st-gcn`:

```powershell
uv run python ..\src\draw_skeleton_mediapipe.py
```

Archivos generados:

- `src/gesto_mano_skeleton.mp4` (video original + esqueleto)
- `src/gesto_mano_skeleton_black.mp4` (fondo negro + esqueleto)
- `src/gesto_mano_preview.gif` (vista previa corta)

Validación:

- Video procesado correctamente
- 93 frames procesados

## 9) Visualización en video con múltiples manos (hasta 8)

Video utilizado:

- `src/otro video/VERT_Applaudissements_à_la_Wikiconvention_Francophone_2022.mp4`

Comando ejecutado desde `st-gcn`:

```powershell
uv run python ..\src\draw_skeleton_mediapipe.py --input "..\src\otro video\VERT_Applaudissements_à_la_Wikiconvention_Francophone_2022.mp4" --num_hands 8
```

Archivos generados:

- `src/otro video/VERT_Applaudissements_à_la_Wikiconvention_Francophone_2022_skeleton.mp4`
- `src/otro video/VERT_Applaudissements_à_la_Wikiconvention_Francophone_2022_skeleton_black.mp4`
- `src/otro video/VERT_Applaudissements_à_la_Wikiconvention_Francophone_2022_preview.gif`

Validación:

- Video procesado correctamente
- 92 frames procesados

## Notas

- `ntu_gendata.py` no aplica a este flujo porque espera directorios con archivos `.skeleton` de NTU, no un `.npy` ya procesado.
