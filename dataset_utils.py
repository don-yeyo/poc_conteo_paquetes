import os
import json
import shutil
import cv2


def convert_yolo_to_coco(yolo_dir, output_dir, classes):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    annotations = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    for i, cls in enumerate(classes):
        annotations['categories'].append({
            'id': i + 1,
            'name': cls
        })
    
    yolo_images_dir = os.path.join(yolo_dir, 'images')
    yolo_labels_dir = os.path.join(yolo_dir, 'labels')
    
    if not os.path.exists(yolo_images_dir):
        print(f"Error: No se encontró {yolo_images_dir}")
        return None
    
    if not os.path.exists(yolo_labels_dir):
        print(f"Advertencia: No se encontró {yolo_labels_dir}")
    
    image_files = [f for f in os.listdir(yolo_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    ann_id = 1
    for img_id, img_file in enumerate(image_files, 1):
        src_path = os.path.join(yolo_images_dir, img_file)
        
        img = cv2.imread(src_path)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        dest_filename = f'image_{img_id:04d}.jpg'
        dest_path = os.path.join(images_dir, dest_filename)
        cv2.imwrite(dest_path, img)
        
        annotations['images'].append({
            'id': img_id,
            'file_name': dest_filename,
            'width': w,
            'height': h
        })
        
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(yolo_labels_dir, label_file)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        
                        x = int((cx - bw / 2) * w)
                        y = int((cy - bh / 2) * h)
                        bw = int(bw * w)
                        bh = int(bh * h)
                        
                        annotations['annotations'].append({
                            'id': ann_id,
                            'image_id': img_id,
                            'category_id': class_id + 1,
                            'bbox': [x, y, bw, bh],
                            'area': bw * bh,
                            'iscrowd': 0
                        })
                        ann_id += 1
    
    labels_path = os.path.join(output_dir, 'labels.json')
    with open(labels_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Convertido {len(annotations['images'])} imágenes")
    print(f"Convertidas {len(annotations['annotations'])} anotaciones")
    print(f"Guardado en: {labels_path}")
    
    return output_dir


def convert_voc_to_coco(voc_dir, output_dir, classes):
    import xml.etree.ElementTree as ET
    
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    annotations = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    for i, cls in enumerate(classes):
        annotations['categories'].append({
            'id': i + 1,
            'name': cls
        })
    
    class_map = {cls: i + 1 for i, cls in enumerate(classes)}
    
    ann_id = 1
    
    jpeg_dir = os.path.join(voc_dir, 'JPEGImages')
    annotation_dir = os.path.join(voc_dir, 'Annotations')
    
    if not os.path.exists(jpeg_dir):
        print(f"Error: No se encontró {jpeg_dir}")
        return None
    
    xml_files = os.listdir(annotation_dir) if os.path.exists(annotation_dir) else []
    
    for img_id, xml_file in enumerate(xml_files, 1):
        if not xml_file.endswith('.xml'):
            continue
        
        xml_path = os.path.join(annotation_dir, xml_file)
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except:
            continue
        
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        img_filename = root.find('filename').text
        src_path = os.path.join(jpeg_dir, img_filename)
        
        img = cv2.imread(src_path)
        if img is None:
            continue
        
        dest_filename = f'image_{img_id:04d}.jpg'
        dest_path = os.path.join(images_dir, dest_filename)
        cv2.imwrite(dest_path, img)
        
        annotations['images'].append({
            'id': img_id,
            'file_name': dest_filename,
            'width': w,
            'height': h
        })
        
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in class_map:
                continue
            
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            bw = xmax - xmin
            bh = ymax - ymin
            
            annotations['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': class_map[cls_name],
                'bbox': [xmin, ymin, bw, bh],
                'area': bw * bh,
                'iscrowd': 0
            })
            ann_id += 1
    
    labels_path = os.path.join(output_dir, 'labels.json')
    with open(labels_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Convertido {len(annotations['images'])} imágenes")
    print(f"Convertidas {len(annotations['annotations'])} anotaciones")
    
    return output_dir


def validate_dataset(dataset_dir):
    labels_file = os.path.join(dataset_dir, 'labels.json')
    
    if not os.path.exists(labels_file):
        print(f"Error: No se encontró labels.json en {dataset_dir}")
        return False
    
    with open(labels_file, 'r') as f:
        annotations = json.load(f)
    
    print(f"Validando dataset: {dataset_dir}")
    print(f"  Imágenes: {len(annotations['images'])}")
    print(f"  Anotaciones: {len(annotations['annotations'])}")
    print(f"  Categorías: {len(annotations['categories'])}")
    
    images_dir = os.path.join(dataset_dir, 'images')
    missing = []
    
    for img in annotations['images']:
        img_path = os.path.join(images_dir, img['file_name'])
        if not os.path.exists(img_path):
            missing.append(img['file_name'])
    
    if missing:
        print(f"\n¡Advertencia! {len(missing)} imágenes faltantes:")
        for m in missing[:5]:
            print(f"  - {m}")
    else:
        print("\n✓ Todas las imágenes presentes")
    
    print("\nCategorías disponibles:")
    for cat in annotations['categories']:
        count = sum(1 for ann in annotations['annotations'] if ann['category_id'] == cat['id'])
        print(f"  - {cat['name']}: {count} anotaciones")
    
    return len(missing) == 0


def split_dataset(dataset_dir, train_ratio=0.8):
    labels_file = os.path.join(dataset_dir, 'labels.json')
    
    with open(labels_file, 'r') as f:
        annotations = json.load(f)
    
    import random
    random.seed(42)
    
    images = annotations['images']
    random.shuffle(images)
    
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    train_ids = {img['id'] for img in train_images}
    
    train_annotations = {
        'images': train_images,
        'annotations': [a for a in annotations['annotations'] if a['image_id'] in train_ids],
        'categories': annotations['categories']
    }
    
    val_annotations = {
        'images': val_images,
        'annotations': [a for a in annotations['annotations'] if a['image_id'] not in train_ids],
        'categories': annotations['categories']
    }
    
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    
    for img in train_images:
        src = os.path.join(dataset_dir, 'images', img['file_name'])
        dst = os.path.join(train_dir, 'images', img['file_name'])
        if os.path.exists(src):
            shutil.copy2(src, dst)
    
    for img in val_images:
        src = os.path.join(dataset_dir, 'images', img['file_name'])
        dst = os.path.join(val_dir, 'images', img['file_name'])
        if os.path.exists(src):
            shutil.copy2(src, dst)
    
    with open(os.path.join(train_dir, 'labels.json'), 'w') as f:
        json.dump(train_annotations, f, indent=2)
    
    with open(os.path.join(val_dir, 'labels.json'), 'w') as f:
        json.dump(val_annotations, f, indent=2)
    
    print(f"Dataset dividido:")
    print(f"  Train: {len(train_images)} imágenes")
    print(f"  Val: {len(val_images)} imágenes")
    print(f"  Guardado en: {train_dir}, {val_dir}")
    
    return train_dir, val_dir


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Utilidades para dataset')
    parser.add_argument('command', choices=['validate', 'split', 'convert-yolo', 'convert-voc'])
    parser.add_argument('--dir', default='dataset', help='Directorio del dataset')
    parser.add_argument('--output', help='Directorio de salida')
    parser.add_argument('--classes', nargs='+', help='Lista de clases')
    args = parser.parse_args()
    
    if args.command == 'validate':
        validate_dataset(args.dir)
    elif args.command == 'split':
        split_dataset(args.dir)
    elif args.command == 'convert-yolo':
        if not args.output or not args.classes:
            print("Error: --output y --classes son requeridos")
        else:
            convert_yolo_to_coco(args.dir, args.output, args.classes)
    elif args.command == 'convert-voc':
        if not args.output or not args.classes:
            print("Error: --output y --classes son requeridos")
        else:
            convert_voc_to_coco(args.dir, args.output, args.classes)