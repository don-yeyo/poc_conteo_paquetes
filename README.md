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

## Instalación

```bash
cd poc_conteo_paquetes
pip install -r requirements.txt
```

La primera ejecución descargará automáticamente el modelo EfficientDet-Lite0 (~4MB).

## Ejecución

```bash
python app.py
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

## Próximos Pasos para Producción

1. Entrenar modelo personalizado para tipo específico de paquetes
2. Agregar base de datos para persistir conteos
3. Implementar API REST para integración
4. Agregar logging y exportación de datos
5. Soporte para múltiples áreas de conteo
6. Alertas visuales/sonoras cuando hay cambios