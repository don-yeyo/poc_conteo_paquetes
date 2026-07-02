import os
import sys
import argparse
from unittest.mock import MagicMock

# Mock array_record to avoid Windows import errors (C++ extension not supported on Windows)
mock_ar = MagicMock()
sys.modules['array_record'] = mock_ar
sys.modules['array_record.python'] = mock_ar
sys.modules['array_record.python.array_record_module'] = mock_ar

from mediapipe_model_maker.object_detector import Dataset, SupportedModels, HParams, ObjectDetector


def train_model(
    dataset_dir,
    output_dir='models',
    model_name='efficientdet_lite0',
    batch_size=2,
    epochs=50,
    learning_rate=0.3,
    shuffle=True,
    validation_split=0.2
):
    print("=" * 60)
    print("ENTRENAMIENTO DE MODELO PERSONALIZADO")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCargando dataset desde: {dataset_dir}")
    print(f"Modelo base: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Épocas: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Validation split: {validation_split}")
    
    spec = SupportedModels.EFFICIENTDET_LITE0
    
    hparams = HParams(
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        shuffle=shuffle,
        batch_denorm=255
    )
    
    print("\nCargando dataset...")
    try:
        dataset = Dataset.from_coco_folder(
            data_dir=dataset_dir,
            max_num_images=None
        )
        print(f"Dataset cargado: {dataset.num_samples} imágenes")
    except Exception as e:
        print(f"Error al cargar dataset: {e}")
        print("\nAsegúrate de que el directorio tenga la estructura:")
        print("  dataset/")
        print("    images/")
        print("      imagen1.jpg")
        print("      ...")
        print("    labels.json")
        return None
    
    print("\nEntrenando modelo...")
    print("Esto puede tomar varios minutos...\n")
    
    model = ObjectDetector.create_from_dataset(
        spec,
        dataset,
        hparams
    )
    
    print("\nEvaluar modelo...")
    loss = model.evaluate(dataset)
    print(f"Pérdida (loss): {loss}")
    
    model_path = os.path.join(output_dir, 'custom_model.tflite')
    print(f"\nExportando modelo a: {model_path}")
    model.export(model_path)
    
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Modelo guardado en: {model_path}")
    print("\nPara usar el modelo en app.py:")
    print(f"  model_path = '{model_path}'")
    print("  detector = ObjectDetector(model_path)")
    
    return model_path


def quick_train(dataset_dir, output_dir='models'):
    return train_model(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        model_name='efficientdet_lite0',
        batch_size=2,
        epochs=25,
        learning_rate=0.3
    )


def evaluate_model(model_path, dataset_dir):
    print("=" * 60)
    print("EVALUACIÓN DE MODELO")
    print("=" * 60)
    
    spec = SupportedModels.EFFICIENTDET_LITE0
    dataset = Dataset.from_coco_folder(data_dir=dataset_dir)
    
    model = ObjectDetector.create_from_model_path(model_path)
    
    print(f"\nDataset: {dataset.num_samples} imágenes")
    loss = model.evaluate(dataset)
    print(f"\nPérdida: {loss}")
    
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar modelo de detección de objetos')
    parser.add_argument('--dataset', type=str, default='dataset', 
                       help='Directorio del dataset (formato COCO)')
    parser.add_argument('--output', type=str, default='models',
                       help='Directorio de salida para el modelo')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Tamaño del batch')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de épocas')
    parser.add_argument('--learning_rate', type=float, default=0.3,
                       help='Learning rate')
    parser.add_argument('--quick', action='store_true',
                       help='Modo rápido (25 épocas)')
    
    args = parser.parse_args()
    
    if args.quick:
        train_model(
            dataset_dir=args.dataset,
            output_dir=args.output,
            batch_size=args.batch_size,
            epochs=25,
            learning_rate=args.learning_rate
        )
    else:
        train_model(
            dataset_dir=args.dataset,
            output_dir=args.output,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )