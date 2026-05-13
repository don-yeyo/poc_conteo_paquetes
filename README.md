# POC - Contabilidad de Paquetes en Tiempo Real

Sistema de detección y conteo de paquetes en video usando MediaPipe y OpenCV.

## Estructura del Proyecto

```
poc_conteo_paquetes/
├── app.py                    # Punto de entrada
├── requirements.txt          # Dependencias
├── src/
│   ├── __init__.py
│   ├── camera_selector.py    # Detección y selección de cámaras
│   ├── object_detector.py    # Detección de objetos con MediaPipe
│   ├── tracker.py            # Tracking de objetos y conteo
│   └── ui.py                 # Interfaz visual y panel de conteo
└── models/                   # Directorio para modelos (vacío)
```

## Requisitos

- Python 3.8+
- Webcam o cámara IP

## Instalación

```bash
cd poc_conteo_paquetes
pip install -r requirements.txt
```

## Ejecución

```bash
python app.py
```

## Características

1. **Detección automática de cámaras**: Escanea y muestra las cámaras disponibles
2. **Selección de cámara**: Interfaz gráfica para elegir la cámara
3. **Detección de objetos**: Usa MediaPipe Object Detector con modelo EfficientDet-Lite0
4. **Tracking**: Sigue objetos entre frames para evitar duplicados
5. **Conteo**: Registra objetos que cruzan una línea de conteo configurable
6. **Panel lateral**: Muestra en tiempo real:
   - Paquetes entrantes
   - Paquetes salientes
   - Total
   - Objetos activos
   - FPS

## Controles

| Tecla | Acción |
|-------|--------|
| Q | Salir del programa |
| R | Resetear contador |
| L | Activar modo ajuste de línea (clic para posicionar) |
| 1-9 | Seleccionar cámara en menú |

## Modo Demo (sin modelo)

Si no tienes el modelo `.tflite`, el sistema usará **Background Subtraction** de OpenCV para detectar movimiento como alternativa.

## Próximos Pasos para Producción

1. Entrenar modelo personalizado para tipo específico de paquetes
2. Configurar zona de interés (ROI) para la caja/área de conteo
3. Agregar base de datos para persistir conteos
4. Implementar API REST para integración
5. Agregar logging y métricas