import cv2
import json
import os
import numpy as np


class LabelingTool:
    def __init__(self, output_dir='dataset', classes=None):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, 'images')
        self.annotations = {'images': [], 'annotations': [], 'categories': []}
        self.next_image_id = 1
        self.next_ann_id = 1
        self.current_image = None
        self.current_image_path = None
        self.current_image_id = None
        self.boxes = []
        self.current_box = None
        self.drawing = False
        self.start_point = None
        self.current_class = 0
        
        if classes:
            self.classes = classes
        else:
            self.classes = ['producto']
        
        for i, cls in enumerate(self.classes):
            self.annotations['categories'].append({
                'id': i + 1,
                'name': cls
            })
        
        os.makedirs(self.images_dir, exist_ok=True)
        self._load_existing()
    
    def _load_existing(self):
        labels_file = os.path.join(self.output_dir, 'labels.json')
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                self.annotations = json.load(f)
                self.next_image_id = max([img['id'] for img in self.annotations['images']]) + 1 if self.annotations['images'] else 1
                self.next_ann_id = max([ann['id'] for ann in self.annotations['annotations']]) + 1 if self.annotations['annotations'] else 1
    
    def save_labels(self):
        labels_file = os.path.join(self.output_dir, 'labels.json')
        with open(labels_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        print(f"Anotaciones guardadas en {labels_file}")
    
    def mouse_callback(self, event, x, y, flags, param):
        if self.current_image_id is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.drawing = True
            self.current_box = {'start': self.start_point, 'end': (x, y)}
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_box['end'] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = self.start_point
            x2, y2 = self.current_box['end']
            bbox = [
                min(x1, x2),
                min(y1, y2),
                abs(x2 - x1),
                abs(y2 - y1)
            ]
            if bbox[2] > 5 and bbox[3] > 5:
                self.boxes.append({
                    'bbox': bbox,
                    'class_id': self.current_class
                })
            self.current_box = None
    
    def add_image_from_camera(self, frame):
        filename = f'image_{self.next_image_id:04d}.jpg'
        filepath = os.path.join(self.images_dir, filename)
        cv2.imwrite(filepath, frame)
        
        h, w = frame.shape[:2]
        self.annotations['images'].append({
            'id': self.next_image_id,
            'file_name': filename,
            'width': w,
            'height': h
        })
        
        self.current_image_id = self.next_image_id
        self.current_image = frame.copy()
        self.current_image_path = filepath
        self.next_image_id += 1
        self.boxes = []
        
        return filename
    
    def add_image_from_file(self, filepath):
        filename = os.path.basename(filepath)
        dest_path = os.path.join(self.images_dir, filename)
        
        img = cv2.imread(filepath)
        if img is None:
            return None
        
        cv2.imwrite(dest_path, img)
        h, w = img.shape[:2]
        
        self.annotations['images'].append({
            'id': self.next_image_id,
            'file_name': filename,
            'width': w,
            'height': h
        })
        
        self.current_image_id = self.next_image_id
        self.current_image_path = dest_path
        self.next_image_id += 1
        self.boxes = []
        
        return filename
    
    def confirm_boxes(self):
        if not self.boxes:
            return 0
        
        for box in self.boxes:
            x, y, w, h = box['bbox']
            self.annotations['annotations'].append({
                'id': self.next_ann_id,
                'image_id': self.current_image_id,
                'category_id': box['class_id'] + 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0
            })
            self.next_ann_id += 1
        
        count = len(self.boxes)
        self.boxes = []
        self.save_labels()
        return count

    def cancel_current_image(self):
        if self.current_image_id is None:
            return
        # Find the image entry in self.annotations['images']
        img_entry = next((img for img in self.annotations['images'] if img['id'] == self.current_image_id), None)
        if img_entry:
            self.annotations['images'].remove(img_entry)
            filepath = os.path.join(self.images_dir, img_entry['file_name'])
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Error al eliminar imagen cancelada: {e}")
        self.boxes = []
        self.current_image_id = None
        print("Captura descartada. Volviendo a vista en vivo.")
    
    def get_current_image(self):
        if self.current_image_path and os.path.exists(self.current_image_path):
            return cv2.imread(self.current_image_path)
        return None
    
    def draw_boxes(self, frame):
        for box in self.boxes:
            x, y, w, h = map(int, box['bbox'])
            class_id = box['class_id']
            color = (0, 255 * (class_id + 1) // len(self.classes), 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, self.classes[class_id], (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if self.current_box:
            x1, y1 = self.current_box['start']
            x2, y2 = self.current_box['end']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        return frame
    
    def set_class(self, class_id):
        self.current_class = class_id
    
    def undo_last_box(self):
        if self.boxes:
            self.boxes.pop()
            return True
        return False
    
    def get_dataset_info(self):
        return {
            'num_images': len(self.annotations['images']),
            'num_annotations': len(self.annotations['annotations']),
            'classes': self.classes
        }


def create_labeling_panel(h, tool, classes):
    pw, ph = 280, h
    panel = np.zeros((ph, pw, 3), dtype=np.uint8)
    panel[:] = (25, 25, 35) # Fondo gris oscuro
    
    y = 25
    cv2.putText(panel, "ETIQUETADO DE IMAGENES", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.line(panel, (10, y + 10), (pw - 10, y + 10), (80, 80, 80), 1)
    y += 40
    
    # Indicador de Modo
    is_live = tool.current_image_id is None
    if is_live:
        mode_text = "VISTA EN VIVO"
        mode_color = (0, 255, 255) # Amarillo/Celeste
    else:
        mode_text = "MODO ETIQUETADO"
        mode_color = (0, 255, 0) # Verde
        
    cv2.putText(panel, "ESTADO:", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(panel, mode_text, (15, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
    y += 50
    
    # Estadísticas
    info = tool.get_dataset_info()
    cv2.putText(panel, f"Total fotos: {info['num_images']}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y += 20
    cv2.putText(panel, f"Total cajas: {info['num_annotations']}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y += 20
    if not is_live:
        cv2.putText(panel, f"Cajas en esta foto: {len(tool.boxes)}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        y += 20
        
    cv2.line(panel, (10, y + 5), (pw - 10, y + 5), (80, 80, 80), 1)
    y += 25
    
    # Lista de clases
    cv2.putText(panel, "CLASES DISPONIBLES:", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    y += 20
    for idx, cls in enumerate(classes):
        color = (0, 255, 0) if idx == tool.current_class else (180, 180, 180)
        prefix = "-> " if idx == tool.current_class else "   "
        text = f"{prefix}[{idx+1}] {cls}"
        cv2.putText(panel, text, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y += 20
        
    cv2.line(panel, (10, y + 5), (pw - 10, y + 5), (80, 80, 80), 1)
    y += 25
    
    # Controles según el modo
    cv2.putText(panel, "CONTROLES:", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    y += 20
    
    if is_live:
        cv2.putText(panel, "[ESPACIO] Capturar Foto", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        y += 20
        cv2.putText(panel, "[S] Guardar y Salir", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        y += 20
        cv2.putText(panel, "[Q / ESC] Salir", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    else:
        cv2.putText(panel, "[ENTER] Guardar Cajas", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        y += 20
        cv2.putText(panel, "[Z] Deshacer Caja", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        y += 20
        cv2.putText(panel, "[C / ESC] Descartar Foto", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        y += 20
        cv2.putText(panel, "[1-9] Cambiar clase", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
                    
    return panel


def run_labeling_interface(classes=None, camera_idx=0):
    if classes is None:
        classes = ['producto']
    
    tool = LabelingTool(classes=classes)
    
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return
    
    window_name = 'Herramienta de Etiquetado'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, tool.mouse_callback)
    
    print("\n" + "=" * 50)
    print("HERRAMIENTA DE ETIQUETADO INICIADA")
    print("Mira el panel lateral de la ventana para ver los controles.")
    print("=" * 50)
    
    while True:
        is_live = tool.current_image_id is None
        if is_live:
            ret, frame = cap.read()
            if not ret:
                break
            display = frame.copy()
        else:
            display = tool.current_image.copy()
            
        display = tool.draw_boxes(display)
        
        h, w = display.shape[:2]
        panel = create_labeling_panel(h, tool, classes)
        combined = np.hstack([display, panel])
        
        cv2.imshow(window_name, combined)
        
        # Check if the window was closed (clicking X)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Ventana cerrada por el usuario.")
            tool.save_labels()
            break

        key = cv2.waitKey(1) & 0xFF
        if is_live:
            if key == ord(' '):
                tool.add_image_from_camera(frame)
                print(f"Imagen {tool.next_image_id - 1} capturada. Dibuje las cajas en la imagen estática.")
            elif key == ord('s') or key == ord('S') or key == ord('q') or key == ord('Q') or key == 27:
                tool.save_labels()
                break
        else:
            if key == 13: # ENTER
                count = tool.confirm_boxes()
                print(f"{count} bounding boxes confirmados.")
            elif key == ord('z') or key == ord('Z'):
                if tool.undo_last_box():
                    print("Última caja eliminada.")
            elif key == ord('c') or key == ord('C') or key == 27: # Cancelar captura
                tool.cancel_current_image()
            elif key >= ord('1') and key <= ord('9'):
                class_idx = key - ord('1')
                if class_idx < len(classes):
                    tool.set_class(class_idx)
                    print(f"Clase seleccionada: {classes[class_idx]}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nProceso de etiquetado terminado")
    print(f"Dataset: {tool.get_dataset_info()}")


def create_dataset_from_folder(image_folder, output_dir='dataset'):
    tool = LabelingTool(output_dir=output_dir)
    
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in extensions:
        image_files.extend([f for f in os.listdir(image_folder) if f.lower().endswith(ext)])
    
    print(f"Encontradas {len(image_files)} imágenes")
    
    for img_file in image_files:
        filepath = os.path.join(image_folder, img_file)
        result = tool.add_image_from_file(filepath)
        if result:
            print(f"Imagen {result} añadida")
    
    tool.save_labels()
    print("\nDataset creado. Ahora usa run_labeling_interface() para etiquetar las imágenes.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Herramienta de etiquetado para detection de objetos')
    parser.add_argument('--classes', nargs='+', default=['producto'], help='Lista de clases')
    parser.add_argument('--camera', type=int, default=0, help='Índice de cámara')
    parser.add_argument('--folder', type=str, help='Carpeta con imágenes para crear dataset')
    args = parser.parse_args()
    
    if args.folder:
        create_dataset_from_folder(args.folder)
    else:
        run_labeling_interface(args.classes, args.camera)