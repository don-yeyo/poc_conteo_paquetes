# POC - Conteo de Paquetes en Cajón

Sistema de detección y conteo de paquetes en tiempo real usando MediaPipe y OpenCV. Diseñado para cámaras posicionadas desde arriba enfocando un cajón donde se agregan/retiran productos.

## Estructura del Proyecto

```
poc_conteo_paquetes/
├── app.py                    # Aplicación principal
├── requirements.txt          # Dependencias
├── models/                   # Modelos TensorFlow Lite (descargados automáticamente)
└── README.md                 # Este archivo
```

## Requisitos

- Python 3.10+
- Webcam o cámara IP
- Windows/Linux/Mac

## Instalación y Configuración del Entorno (Windows)

Dado que TensorFlow y MediaPipe Model Maker tienen restricciones estrictas de versión (no soportan Python 3.13 o 3.14 y requieren dependencias específicas de C++), se recomienda utilizar un entorno virtual con **Python 3.10**.

1. **Crear el entorno virtual con Python 3.10** (si utilizas `uv`):
   ```bash
   uv venv --python 3.10
   ```

2. **Activar el entorno virtual**:
   - En Windows (PowerShell):
     ```powershell
     .venv\Scripts\activate
     ```
   - En Windows (CMD):
     ```cmd
     .venv\Scripts\activate.bat
     ```

3. **Instalar dependencias**:
   ```bash
   uv pip install -r requirements.txt
   ```
   *Nota: Si utilizas pip tradicional en lugar de uv, instala `cython<3.0.0` y `wheel` antes de instalar los requisitos para evitar errores de compilación de `pyyaml`.*

La primera ejecución descargará automáticamente el modelo EfficientDet-Lite0 (~4MB).

## Ejecución

Para iniciar la aplicación usando el entorno virtual configurado:

```bash
.venv\Scripts\python app.py
```

## Características

1. **Detección automática de cámaras**: Escanea las cámaras disponibles automáticamente
2. **Detección de objetos**: Usa MediaPipe Object Detector con modelo EfficientDet-Lite0
3. **Área de conteo configurable**: Define un rectángulo sobre el cajón con el mouse
4. **Tracking de objetos**: Sigue objetos entre frames con ID único
5. **Conteo bidireccional**: Detecta cuando objetos ingresan o salen del área
6. **Panel lateral**: Muestra en tiempo real:
   - Paquetes que ingresaron
   - Paquetes que salieron
   - Total neto
   - Estado del área definida
   - Objetos activos detectados
   - FPS

## Controles

| Tecla | Acción |
|-------|--------|
| **L** | Activar modo definición de área (clic y arrastra para dibujar el cajón) |
| **R** | Resetear conteo (reinicia contadores) |
| **Q** | Salir del programa |

## Cómo Usar

1. **Ejecutar la aplicación**: `python app.py`
2. **Definir el área del cajón**:
   - Presiona **L** para activar el modo de definición
   - Clic y arrastra para dibujar un rectángulo sobre el cajón en la imagen
   - El área se marcará en verde
3. **Observar el conteo**:
   - Los objetos detectados se marcan con recuadros verdes
   - Cada objeto tiene un ID único
   - La consola muestra cuando un paquete ingresa o sale del área
4. **Ajustes**:
   - Presiona **R** para resetear los contadores si es necesario
   - Presiona **L** de nuevo para redefinir el área

## Ejemplo de Salida en Consola

```
Camaras: [0]
Descargando modelo...
100%
ROI actualizado: x=101-265, y=30-241
>>> PAQUETE INGRESÓ #1 - ID:12 en (221,176)
>>> PAQUETE SALIÓ #1 - ID:13 de (340,200)
>>> PAQUETE INGRESÓ #2 - ID:15 en (180,150)
```

## Consideraciones para Optimizar

1. **Iluminación**: Usar iluminación uniforme sobre el cajón
2. **Contraste**: Los paquetes deben verse claramente contra el fondo
3. **Tamaño mínimo**: Ajustar `score_threshold` si hay muchos falsos positivos
4. **Perspectiva**: La cámara debe estar perpendicular al cajón para mejor detección

## Tecnologías

- **MediaPipe**: Detección de objetos en tiempo real
- **OpenCV**: Procesamiento de video y visualizaciones
- **TensorFlow Lite**: Modelo EfficientDet-Lite0 para detección
- **NumPy**: Manipulación de arrays

## Entrenamiento de Modelo Personalizado

Esta aplicación incluye herramientas para entrenar un modelo de detección específico para tus productos.

### Estructura de Archivos de Entrenamiento

```
poc_conteo_paquetes/
├── app.py                 # Aplicación principal con menú
├── labeling_tool.py       # Herramienta de etiquetado de imágenes
├── train_model.py        # Script de entrenamiento
├── dataset_utils.py      # Utilidades para datasets
├── dataset/              # Dataset (se crea automáticamente)
│   ├── images/           # Imágenes etiquetadas
│   └── labels.json       # Anotaciones en formato COCO
├── models/               # Modelos entrenados
│   ├── efficientdet_lite0.tflite  # Modelo por defecto
│   └── custom_model.tflite        # Tu modelo personalizado
└── README.md
```

---

## Flujo Completo de Entrenamiento

### Paso 1: Recopilar Imágenes

```
1. Ejecutar: python app.py
2. Seleccionar opción 2 (Etiquetar imágenes)
3. Usar la cámara para capturar fotos de tus productos
4. Dibujar bounding boxes alrededor de cada producto
5. Confirmar con ENTER, guardar con S
```

**Recomendaciones:**
- Captura 50-100 imágenes por tipo de producto
- Varía ángulos, distancias e iluminación
- Incluye casos negativos (otros objetos no relevantes)

### Paso 2: Etiquetar Imágenes (Opcional - si ya tienes fotos)

Si ya tienes fotos en una carpeta:

```bash
python labeling_tool.py --folder /ruta/a/tus/imagenes
python labeling_tool.py --classes producto1 producto2 producto3
```

Controles de la herramienta de etiquetado:
| Tecla | Acción |
|-------|--------|
| **ESPACIO** | Capturar imagen de cámara |
| **1-9** | Seleccionar clase del producto |
| **ENTER** | Confirmar bounding boxes |
| **Z** | Deshacer último box |
| **S, Q, ESC** o clic en **X** | Guardar anotaciones y salir |

### Paso 3: Entrenar el Modelo

```bash
python train_model.py --dataset dataset --epochs 50
```

Opciones avanzadas:
```bash
python train_model.py --dataset dataset \
    --output models \
    --batch_size 2 \
    --epochs 100 \
    --learning_rate 0.3
```

Parámetros de entrenamiento:
| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--dataset` | dataset | Directorio con imágenes y labels.json |
| `--output` | models | Directorio de salida del modelo |
| `--batch_size` | 2 | Tamaño del batch (menor = menos RAM) |
| `--epochs` | 50 | Número de épocas |
| `--learning_rate` | 0.3 | Tasa de aprendizaje |
| `--quick` | - | Modo rápido (25 épocas) |

### Paso 4: Usar el Modelo Personalizado

Al ejecutar `python app.py`:
1. Seleccionar opción 1 (Modo Detección)
2. Elegir "Modelo personalizado" cuando pregunte

---

## Formato del Dataset (COCO)

El sistema usa formato COCO. Estructura esperada:

```
dataset/
├── images/
│   ├── image_0001.jpg
│   ├── image_0002.jpg
│   └── ...
└── labels.json
```

Ejemplo de `labels.json`:
```json
{
  "images": [
    {"id": 1, "file_name": "image_0001.jpg", "width": 640, "height": 480}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 50, 80, 120], "area": 9600, "iscrowd": 0}
  ],
  "categories": [
    {"id": 1, "name": "producto1"},
    {"id": 2, "name": "producto2"}
  ]
}
```

---

## Utilidades de Dataset

### Convertir desde YOLO

```bash
python dataset_utils.py convert-yolo --dir /ruta/yolo --output dataset --classes producto1 producto2
```

### Convertir desde Pascal VOC

```bash
python dataset_utils.py convert-voc --dir /ruta/VOC --output dataset --classes producto1 producto2
```

### Validar Dataset

```bash
python dataset_utils.py validate --dir dataset
```

### Dividir Train/Val

```bash
python dataset_utils.py split --dir dataset
```

---

## Menú Principal (app.py)

```
==================================================
MENU PRINCIPAL - Conteo de Paquetes
==================================================
  1 - Modo Detección (usar modelo existente)
  2 - Etiquetar nuevas imágenes
  3 - Entrenar modelo personalizado
  4 - Salir
==================================================
```

| Opción | Descripción |
|--------|-------------|
| 1 | Ejecuta la aplicación de detección en tiempo real |
| 2 | Abre la herramienta de etiquetado para crear dataset |
| 3 | Inicia el entrenamiento con Model Maker |
| 4 | Cierra la aplicación |

---

## Controles en Modo Detección

| Tecla | Acción |
|-------|--------|
| **L** | Activar modo definición de área (clic y arrastra para dibujar el cajón) |
| **R** | Resetear conteo (reinicia contadores) |
| **C** | Cambiar cámara (si hay múltiples) |
| **Q** | Salir del programa |

---

## Solución de Problemas

### Error al cargar dataset
```
ValueError: If the input data directory is empty
```
→ Verifica que `dataset/images/` tenga imágenes y que `dataset/labels.json` exista

### Error de memoria
```
OOM when allocating tensor
```
→ Reduce `--batch_size` a 1

### El modelo no detecta bien
→ Necesitas más imágenes de entrenamiento (mínimo 50 por clase)

### El modelo detecta objetos no deseados
→ Agrega más imágenes negativas al dataset y aumenta el `score_threshold` en app.py

---

## Próximos Pasos para Producción

1. ✅ Entrenar modelo personalizado para tipo específico de paquetes
2. Agregar base de datos para persistir conteos
3. Implementar API REST para integración
4. Agregar logging y exportación de datos
5. Soporte para múltiples áreas de conteo
6. Alertas visuales/sonoras cuando hay cambios