# CONTEXTO DEL PROYECTO - POC Conteo de Paquetes

> Última actualización: 18 Mayo 2026
> Rama: `main` (up to date with origin/main)

---

## ¿QUÉ HACE ESTE PROYECTO?

Sistema de detección y conteo de paquetes en tiempo real usando cámara. El usuario define un área (el cajón) con clic y arrastre, y el sistema cuenta cuántos paquetes entran/salen de esa área usando detección de objetos con MediaPipe + EfficientDet-Lite0.

---

## ESTRUCTURA DEL PROYECTO

```
poc_conteo_paquetes/
├── app.py                 # Aplicación principal con menú y detección
├── labeling_tool.py       # Herramienta de etiquetado de imágenes
├── train_model.py         # Entrenamiento de modelo personalizado
├── dataset_utils.py       # Utilidades: convertir YOLO/VOC, validar, dividir
├── requirements.txt       # opencv-python, mediapipe
├── README.md              # Documentación (MODIFICADO, sin commit)
├── contexto.md            # ← ESTE ARCHIVO
├── dataset/
│   ├── images/            # 24 imágenes capturadas (image_0001.jpg - image_0024.jpg)
│   └── labels.json        # Anotaciones COCO (15 anotaciones, PROBS: image_id=null)
├── models/
│   └── efficientdet_lite0.tflite  # Modelo base descargado (~4MB)
└── __pycache__/
```

---

## ARCHIVOS Y SU ROL

### `app.py` (MODIFICADO, sin commit)
Archivo principal. Contiene:
- **`ObjectDetector`**: Wrapper de MediaPipe Object Detector. Usa EfficientDet-Lite0. Soportado por defecto y modelo personalizado.
- **`ObjectTracker`**: Tracking de objetos entre frames usando IoU. Asigna IDs únicos. Configurable: `max_disappeared=40`, `iou_threshold=0.12`.
- **`AreaCounter`**: Lógica de conteo bidireccional. Detecta transiciones dentro/fuera del ROI.
- **`ROISelector`**: Interfaz para definir el área del cajón con clic y arrastre.
- **`main()`**: Menú principal con 4 opciones: 1-Detección, 2-Etiquetar, 3-Entrenar, 4-Salir.
- **`download_model()`**: Descarga EfficientDet-Lite0 si no existe.
- **`get_cameras()`**: Escanea hasta 5 cámaras.
- **`create_panel()`**: Panel lateral de stats (ingresaron, salieron, neto, FPS, objetos activos).
- **`draw_objects()`**: Dibuja bounding boxes, IDs y trayectorias.

### `labeling_tool.py` (UNTRACKED)
Herramienta de etiquetado con cámara. Controles: ESPACIO=capturar, 1-9=clase, ENTER=confirmar boxes, Z=deshacer, S=guardar. Guarda en formato COCO.

### `train_model.py` (UNTRACKED)
Entrenamiento usando MediaPipe Model Maker. Carga dataset COCO, entrena EfficientDet-Lite0 transfer learning, exporta `custom_model.tflite`.

### `dataset_utils.py` (UNTRACKED)
Utilidades: `convert_yolo_to_coco()`, `convert_voc_to_coco()`, `validate_dataset()`, `split_dataset()`.

---

## ESTADO ACTUAL (git status)

### Commits existentes (3):
- `7f29bad` - "multi camara funcionando" (HEAD)
- `3d679ec` - "unificando"
- `e8b456c` - "funciona deteccion de objetos"

### Cambios sin commit:
- **`app.py`**: Modificado (se le agregó menú principal multicámara, labeling_tool, y entrenamiento)
- **`README.md`**: Modificado (documentación expandida)
- **Archivos untracked**: `dataset/`, `dataset_utils.py`, `labeling_tool.py`, `train_model.py`, `__pycache__/`

### NOTA IMPORTANTE:
Los archivos `dataset_utils.py`, `labeling_tool.py`, `train_model.py` y la carpeta `dataset/` fueron añadidos DESPUÉS del último commit. El dataset tiene imágenes capturadas (`dataset/images/`) pero **labels.json tiene `image_id: null` en todas las anotaciones**, lo que indica que las anotaciones no están correctamente vinculadas a las imágenes.

---

## PROBLEMAS CONOCIDOS / PENDIENTES

### 1. labels.json corrupto
Las anotaciones en `dataset/labels.json` tienen `"image_id": null` en lugar del ID de imagen correspondiente. Esto hace que el entrenamiento falle. Hay que re-etiquetar o reparar este archivo.

### 2. Sin modelo personalizado entrenado
Solo existe `efficientdet_lite0.tflite` (modelo genérico). No hay `custom_model.tflite`. El modelo genérico detecta objetos COCO (personas, sillas, etc.) pero no productos específicos.

### 3. Dependencia de TensorFlow para entrenar
`train_model.py` requiere `mediapipe-model-maker` que a su vez necesita TensorFlow. No está en `requirements.txt`. Esto puede dar conflictos de versión.

### 4. Sin test / validación automatizada
No hay tests unitarios ni scripts de validación del pipeline completo.

---

## CÓMO USAR

```bash
# Detección en tiempo real
python app.py
# Menú → opción 1

# Etiquetar imágenes
python app.py
# Menú → opción 2

# Entrenar (desde línea de comandos)
python train_model.py --dataset dataset --epochs 50

# Utilidades de dataset
python dataset_utils.py validate --dir dataset
python dataset_utils.py split --dir dataset
python dataset_utils.py convert-yolo --dir /ruta --output dataset --classes prod1 prod2
python dataset_utils.py convert-voc --dir /ruta --output dataset --classes prod1 prod2

# Labeling tool directa (para etiquetar fotos existentes)
python labeling_tool.py --folder /ruta/a/imagenes
python labeling_tool.py --classes producto1 producto2 producto3
```

---

## DEPENDENCIAS

```txt
opencv-python>=4.8.0
mediapipe>=0.10.35
```

Para entrenar se necesita adicionalmente: `mediapipe-model-maker` (instala TensorFlow automáticamente).

---

## PRÓXIMOS PASOS (de README)

1. ✅ Entrenar modelo personalizado para tipo específico de paquetes
2. ❌ Agregar base de datos para persistir conteos
3. ❌ Implementar API REST para integración
4. ❌ Agregar logging y exportación de datos
5. ❌ Soporte para múltiples áreas de conteo
6. ❌ Alertas visuales/sonoras cuando hay cambios
7. ❌ Reparar labels.json (image_id: null)
8. ❌ Agregar `mediapipe-model-maker` a requirements.txt

---

## ARQUITECTURA TÉCNICA

```
app.py main()
  ├── show_main_menu() → choice 1-4
  ├── [1] → get_cameras() → select_camera_menu()
  │         → download_model() (si usa default)
  │         → ObjectDetector(model_path, max_results=15, score_threshold=0.35)
  │         → ObjectTracker(max_disappeared=40, iou_threshold=0.12)
  │         → AreaCounter()
  │         → ROIselector() (tecla L activa)
  │         → Loop: detect → track → count → draw
  ├── [2] → run_labeling_mode() → labeling_tool.run_labeling_interface()
  └── [3] → run_training_mode() → train_model.train_model()
```

### Pipeline de detección (frame rate):
1. `detector.detect(frame, timestamp_ms)` → detections
2. `tracker.update(detections)` → objects {id: {bbox, score}}
3. `counter.update(objects)` → actualiza count_in/count_out según transiciones ROI
4. `draw_objects(frame, objects, history)` → dibuja resultados
5. `roi_selector.draw_roi(frame)` → dibuja área verde del cajón
6. `create_panel(...)` → panel lateral con stats

### Pipeline de entrenamiento:
1. `Dataset.from_coco_folder(data_dir=dataset_dir)`
2. `ObjectDetector.create_from_dataset(spec, dataset, hparams)`
3. `model.export(output_path)` → `custom_model.tflite`

---

## FLUJO PARA CONTINUAR (si me quedo sin tokens)

Si este archivo lo lee otro modelo:

1. **Reparar dataset primero**: Las anotaciones en `dataset/labels.json` tienen `image_id: null`. Revisa las imágenes, renumera IDs y asigna `image_id` correctamente.
2. **Entrenar modelo**: Usa `python train_model.py --dataset dataset --epochs 50` (o considerar `--quick` para pruebas).
3. **El usuario probablemente pidió algo específico** que no llegué a completar. Revisa el historial de la conversación o pregúntale qué necesita.
4. **Si pide mejoras**: Revisa los "Próximos Pasos" arriba y los "Problemas Conocidos".
5. **Para probar**: `python app.py`, opción 1, seleccionar cámara.
6. **Commit sin hacer**: Los cambios en `app.py` y `README.md` están sin commit, y hay archivos nuevos untracked. Si el usuario pide commit, añadir todo con `git add -A`.
