import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os


class ObjectDetector:
    def __init__(self, model_path=None, max_results=10, score_threshold=0.4):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            max_results=max_results,
            score_threshold=score_threshold
        )
        self.detector = vision.ObjectDetector.create_from_options(options)
    
    def detect(self, frame, timestamp_ms):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.detector.detect_for_video(mp_image, timestamp_ms)


class ObjectTracker:
    def __init__(self, max_disappeared=30, iou_threshold=0.15):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.history = {}
    
    def compute_iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1, area2 = w1 * h1, w2 * h2
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def update(self, detections):
        if not detections or not detections.detections:
            for oid in list(self.objects.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.disappeared[oid]
                    if oid in self.history:
                        del self.history[oid]
            return self.objects
        
        bboxes = [(int(d.bounding_box.origin_x), int(d.bounding_box.origin_y),
                   int(d.bounding_box.width), int(d.bounding_box.height),
                   d.categories[0].score if d.categories else 0)
                  for d in detections.detections]
        
        if not self.objects:
            for bbox in bboxes:
                oid = self.next_id
                self.objects[oid] = {'bbox': bbox[:4], 'score': bbox[4]}
                self.disappeared[oid] = 0
                self.history[oid] = [self.get_center(bbox[:4])]
                self.next_id += 1
            return self.objects
        
        matched = set()
        for oid in list(self.objects.keys()):
            best_iou, best_idx = 0, -1
            for i, bbox in enumerate(bboxes):
                if i in matched:
                    continue
                iou = self.compute_iou(self.objects[oid]['bbox'], bbox[:4])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            
            if best_iou > self.iou_threshold and best_idx >= 0:
                self.objects[oid] = {'bbox': bboxes[best_idx][:4], 'score': bboxes[best_idx][4]}
                self.disappeared[oid] = 0
                center = self.get_center(bboxes[best_idx][:4])
                self.history.setdefault(oid, []).append(center)
                if len(self.history.get(oid, [])) > 50:
                    self.history[oid] = self.history[oid][-50:]
                matched.add(best_idx)
            else:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.disappeared[oid]
                    if oid in self.history:
                        del self.history[oid]
        
        for i, bbox in enumerate(bboxes):
            if i not in matched:
                oid = self.next_id
                self.objects[oid] = {'bbox': bbox[:4], 'score': bbox[4]}
                self.disappeared[oid] = 0
                self.history[oid] = [self.get_center(bbox[:4])]
                self.next_id += 1
        
        return self.objects
    
    def get_center(self, bbox):
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)
    
    def get_history(self, oid):
        return self.history.get(oid, [])
    
    def reset(self):
        self.objects = {}
        self.disappeared = {}
        self.history = {}
        self.next_id = 0


class AreaCounter:
    def __init__(self, roi=None):
        self.roi = roi
        self.count_in = 0
        self.count_out = 0
        self.object_state = {}
    
    def set_roi(self, x1, y1, x2, y2):
        self.roi = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        self.object_state = {}
        print(f"ROI configurado: ({x1},{y1}) -> ({x2},{y2})")
    
    def is_point_in_roi(self, px, py):
        if self.roi is None:
            return False
        x, y, w, h = self.roi
        return x <= px <= x + w and y <= py <= y + h
    
    def update(self, objects):
        if self.roi is None:
            return
        
        print(f"\n--- Frame: {len(objects)} objetos detectados ---")
        
        for oid, data in objects.items():
            x, y, w, h = data['bbox']
            center_x, center_y = x + w // 2, y + h // 2
            
            in_roi = self.is_point_in_roi(center_x, center_y)
            print(f"  ID:{oid} center=({center_x},{center_y}) inside={in_roi}")
            
            if oid not in self.object_state:
                self.object_state[oid] = {'inside': in_roi, 'first_seen': True}
            else:
                was_inside = self.object_state[oid]['inside']
                self.object_state[oid]['inside'] = in_roi
                
                if self.object_state[oid].get('first_seen', False):
                    self.object_state[oid]['first_seen'] = False
                    continue
                
                if not was_inside and in_roi:
                    self.count_in += 1
                    print(f">>> PAQUETE INGRESÓ #{self.count_in} - ID:{oid}")
                elif was_inside and not in_roi:
                    self.count_out += 1
                    print(f">>> PAQUETE SALIÓ #{self.count_out} - ID:{oid}")
        
        for oid in list(self.object_state.keys()):
            if oid not in objects:
                del self.object_state[oid]
    
    def reset(self):
        self.count_in = 0
        self.count_out = 0
        self.object_state = {}
    
    def get_counts(self):
        return {'in': self.count_in, 'out': self.count_out}


class ROIselector:
    def __init__(self):
        self.roi = None
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.confirmed = False
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.end_point = (x, y)
            self.drawing = True
            self.confirmed = False
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.end_point = (x, y)
            self.drawing = False
            self.confirmed = True
            self.roi = (
                min(self.start_point[0], self.end_point[0]),
                min(self.start_point[1], self.end_point[1]),
                abs(self.end_point[0] - self.start_point[0]),
                abs(self.end_point[1] - self.start_point[1])
            )
    
    def reset(self):
        self.roi = None
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.confirmed = False
    
    def get_roi(self):
        return self.roi
    
    def draw_roi(self, frame):
        if self.start_point and self.end_point:
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            
            if self.drawing:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "Dibujando ROI...", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if self.roi:
            x, y, w, h = self.roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, "AREA DE CAJON", (x + 5, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        elif not self.drawing:
            cv2.putText(frame, "Presione L para definir area del cajon", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
        
        return frame


def download_model():
    import urllib.request
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'efficientdet_lite0.tflite')
    
    if os.path.exists(model_path):
        return model_path
    
    url = 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite'
    print(f"Descargando modelo...")
    urllib.request.urlretrieve(url, model_path, reporthook=lambda c, bs, ts: print(f"\r{int(c*bs*100/ts)}%", end='', flush=True))
    print(f"\nModelo descargado")
    return model_path


def create_panel(h, counter, fps, active, roi_set):
    pw, ph = 280, h
    panel = np.zeros((ph, pw, 3), dtype=np.uint8)
    panel[:] = (25, 25, 35)
    
    y = 25
    cv2.putText(panel, "CONTEO DE PAQUETES", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.line(panel, (10, y + 10), (pw - 10, y + 10), (80, 80, 80), 1)
    y += 50
    
    cv2.putText(panel, "INGRESARON:", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(panel, str(counter.count_in), (20, y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)
    y += 80
    
    cv2.putText(panel, "SALIERON:", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)
    cv2.putText(panel, str(counter.count_out), (20, y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 150, 255), 2)
    y += 80
    
    cv2.line(panel, (10, y - 15), (pw - 10, y - 15), (100, 100, 100), 1)
    cv2.putText(panel, f"NETO: {counter.count_in - counter.count_out}", (20, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 45
    
    status = "OK" if roi_set else "NO DEFINIDA"
    color = (0, 255, 0) if roi_set else (0, 0, 255)
    cv2.putText(panel, f"Area cajon: {status}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    y += 30
    
    cv2.putText(panel, f"Objetos activos: {active}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    y += 25
    cv2.putText(panel, f"FPS: {int(fps)}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    
    y = ph - 100
    cv2.line(panel, (10, y), (pw - 10, y), (80, 80, 80), 1)
    y += 15
    cv2.putText(panel, "CONTROLES:", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y += 22
    cv2.putText(panel, "L = Definir area", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y += 18
    cv2.putText(panel, "R = Resetear conteo", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y += 18
    cv2.putText(panel, "Q = Salir", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return panel


def draw_objects(frame, objects, history):
    for oid, data in objects.items():
        x, y, w, h = data['bbox']
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        center_x, center_y = x + w // 2, y + h // 2
        cv2.circle(frame, (center_x, center_y), 3, (0, 255, 255), -1)
        
        label = f"ID:{oid}"
        cv2.putText(frame, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        
        h_oid = history.get(oid, [])
        if len(h_oid) > 1:
            for i in range(1, len(h_oid)):
                cv2.line(frame, h_oid[i-1], h_oid[i], (255, 200, 0), 1)
    
    return frame


def get_cameras(max_check=5):
    cameras = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cameras.append(i)
            cap.release()
    return cameras


def main():
    print("=" * 50)
    print("POC - Conteo de Paquetes en Cajon")
    print("=" * 50)
    
    cameras = get_cameras()
    if not cameras:
        print("No se encontraron camaras")
        return
    
    print(f"Camaras: {cameras}")
    cam_idx = cameras[0]
    
    model_path = download_model()
    
    print("Inicializando detector...")
    detector = ObjectDetector(model_path, max_results=15, score_threshold=0.35)
    tracker = ObjectTracker(max_disappeared=40, iou_threshold=0.12)
    counter = AreaCounter()
    roi_selector = ROIselector()
    
    cap = cv2.VideoCapture(cam_idx)
    ret, frame = cap.read()
    if not ret:
        print("Error al abrir camara")
        return
    
    h, w = frame.shape[:2]
    print(f"Resolucion: {w}x{h}")
    
    print("\nControles:")
    print("  L = Definir area del cajon (clic y arrastrar)")
    print("  R = Resetear conteo")
    print("  Q = Salir")
    
    roi_mode = False
    fps = 0
    fc = 0
    start = time.time()
    frame_timestamp = 0
    frame_interval_ms = 33
    
    window_name = 'POC Conteo Cajon'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, roi_selector.mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_timestamp += frame_interval_ms
        detections = detector.detect(frame, frame_timestamp)
        objects = tracker.update(detections)
        
        if roi_selector.roi:
            counter.set_roi(
                roi_selector.roi[0], roi_selector.roi[1],
                roi_selector.roi[0] + roi_selector.roi[2],
                roi_selector.roi[1] + roi_selector.roi[3]
            )
        
        counter.update(objects)
        
        frame = draw_objects(frame, objects, tracker.history)
        frame = roi_selector.draw_roi(frame)
        
        roi_set = roi_selector.roi is not None
        panel = create_panel(h, counter, fps, len(objects), roi_set)
        combined = np.hstack([frame, panel])
        
        cv2.imshow(window_name, combined)
        
        fc += 1
        if fc >= 10:
            fps = fc / (time.time() - start)
            fc = 0
            start = time.time()
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('r') or key == ord('R'):
            counter.reset()
            tracker.reset()
            print(">>> CONTEO RESETEADO")
        elif key == ord('l') or key == ord('L'):
            roi_mode = not roi_mode
            roi_selector.reset()
            if roi_mode:
                print(">>> Modo definir area: clic y arrastra para marcar el cajon")
            else:
                print(">>> Saliendo del modo definir area")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nResultado: Ingresaron={counter.count_in}, Salieron={counter.count_out}, Neto={counter.count_in - counter.count_out}")


if __name__ == '__main__':
    main()