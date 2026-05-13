import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os


class ObjectDetector:
    def __init__(self, model_path=None, max_results=10, score_threshold=0.5):
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
    def __init__(self, max_disappeared=30, iou_threshold=0.2):
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
                self.history[oid] = [bbox[:4]]
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
                self.history.setdefault(oid, []).append(bboxes[best_idx][:4])
                if len(self.history[oid]) > 30:
                    self.history[oid] = self.history[oid][-30:]
                matched.add(best_idx)
            else:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.history[oid]
        
        for i, bbox in enumerate(bboxes):
            if i not in matched:
                oid = self.next_id
                self.objects[oid] = {'bbox': bbox[:4], 'score': bbox[4]}
                self.disappeared[oid] = 0
                self.history[oid] = [bbox[:4]]
                self.next_id += 1
        
        return self.objects
    
    def get_history(self, oid):
        return self.history.get(oid, [])
    
    def reset(self):
        self.objects = {}
        self.disappeared = {}
        self.history = {}
        self.next_id = 0


class LineCounter:
    def __init__(self, line_y):
        self.line_y = line_y
        self.count_in = 0
        self.count_out = 0
        self.crossed = {}
    
    def check_crossing(self, prev_y, curr_y):
        if prev_y < self.line_y <= curr_y:
            return 'in'
        if prev_y > self.line_y >= curr_y:
            return 'out'
        return None
    
    def update(self, objects):
        for oid, data in objects.items():
            x, y, w, h = data['bbox']
            curr_center_y = y + h // 2
            
            history = self.crossed.get(oid, [])
            
            if len(history) == 0:
                self.crossed[oid] = [curr_center_y]
                continue
            
            prev_y = history[-1]
            
            if oid not in self.crossed:
                self.crossed[oid] = [curr_center_y]
            else:
                self.crossed[oid].append(curr_center_y)
                if len(self.crossed[oid]) > 10:
                    self.crossed[oid] = self.crossed[oid][-10:]
            
            direction = self.check_crossing(prev_y, curr_center_y)
            
            if direction == 'in':
                self.count_in += 1
                print(f">>> PAQUETE ENTRANTE #{self.count_in} - ID:{oid}")
            elif direction == 'out':
                self.count_out += 1
                print(f">>> PAQUETE SALIENTE #{self.count_out} - ID:{oid}")
        
        for oid in list(self.crossed.keys()):
            if oid not in objects:
                del self.crossed[oid]
    
    def reset(self):
        self.count_in = 0
        self.count_out = 0
        self.crossed = {}


def download_model():
    import urllib.request
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'efficientdet_lite0.tflite')
    
    if os.path.exists(model_path):
        print(f"Modelo ya existe: {model_path}")
        return model_path
    
    url = 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite'
    print(f"Descargando modelo desde {url}...")
    
    def progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rProgreso: {percent}%", end='', flush=True)
    
    urllib.request.urlretrieve(url, model_path, reporthook=progress)
    print(f"\nModelo descargado: {model_path}")
    return model_path


def draw_info(frame, objects, history, line_y, counter):
    h, w = frame.shape[:2]
    
    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 3)
    cv2.putText(frame, f"Linea de conteo Y={line_y}", (10, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    for oid, data in objects.items():
        x, y, bw, bh = data['bbox']
        
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        
        label = f"ID:{oid} ({data['score']:.2f})"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        h_oid = history.get(oid, [])
        if len(h_oid) > 1:
            for i in range(1, len(h_oid)):
                p1 = (h_oid[i-1][0] + h_oid[i-1][2]//2, h_oid[i-1][1] + h_oid[i-1][3]//2)
                p2 = (h_oid[i][0] + h_oid[i][2]//2, h_oid[i][1] + h_oid[i][3]//2)
                cv2.line(frame, p1, p2, (255, 255, 0), 1)
    
    return frame


def create_panel(h, counter, fps, active):
    pw, ph = 280, h
    panel = np.zeros((ph, pw, 3), dtype=np.uint8)
    panel[:] = (25, 25, 35)
    
    y = 30
    cv2.putText(panel, "CONTEO PAQUETES", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y += 50
    cv2.putText(panel, "ENTRANTES", (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(panel, str(counter.count_in), (30, y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    y += 80
    cv2.putText(panel, "SALIENTES", (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)
    cv2.putText(panel, str(counter.count_out), (30, y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 150, 255), 2)
    
    y += 80
    cv2.line(panel, (20, y - 10), (pw - 20, y - 10), (100, 100, 100), 1)
    cv2.putText(panel, f"TOTAL: {counter.count_in - counter.count_out}", (30, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    y += 60
    cv2.putText(panel, f"Objetos: {active}", (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(panel, f"FPS: {int(fps)}", (30, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    y += 80
    cv2.line(panel, (20, y - 20), (pw - 20, y - 20), (80, 80, 80), 1)
    cv2.putText(panel, "CONTROLES:", (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(panel, "Q=Salir  R=Reset", (30, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    return panel


def get_cameras():
    cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cameras.append(i)
            cap.release()
    return cameras


def main():
    print("=" * 50)
    print("POC - Conteo de Paquetes con MediaPipe")
    print("=" * 50)
    
    cameras = get_cameras()
    if not cameras:
        print("No se encontraron camaras")
        return
    
    print(f"Camaras disponibles: {cameras}")
    cam_idx = cameras[0] if len(cameras) == 1 else None
    
    if cam_idx is None:
        for i, c in enumerate(cameras):
            print(f"{i+1}. Camara {c}")
        print("Seleccione camara: ", end='')
        sel = int(input()) - 1
        cam_idx = cameras[sel]
    
    print(f"Usando camara {cam_idx}")
    
    model_path = download_model()
    
    print("Inicializando detector...")
    detector = ObjectDetector(model_path, max_results=10, score_threshold=0.4)
    tracker = ObjectTracker(max_disappeared=40, iou_threshold=0.15)
    
    cap = cv2.VideoCapture(cam_idx)
    ret, frame = cap.read()
    if not ret:
        print("Error al abrir camara")
        return
    
    h, w = frame.shape[:2]
    counter = LineCounter(line_y=h // 2)
    
    print("\nIniciando... Presione Q para salir")
    
    fps = 0
    fc = 0
    start = time.time()
    ts = 0
    
    cv2.namedWindow('POC Conteo', cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        ts = int((time.time() - start) * 1000)
        detections = detector.detect(frame, ts)
        objects = tracker.update(detections)
        counter.update(objects)
        
        frame = draw_info(frame, objects, tracker.history, counter.line_y, counter)
        
        panel = create_panel(h, counter, fps, len(objects))
        combined = np.hstack([frame, panel])
        
        cv2.imshow('POC Conteo', combined)
        
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
        elif key == ord('u') or key == ord('U'):
            counter.line_y = max(0, counter.line_y - 10)
        elif key == ord('d') or key == ord('D'):
            counter.line_y = min(h, counter.line_y + 10)
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinalizado. Entrantes: {counter.count_in}, Salientes: {counter.count_out}")


if __name__ == '__main__':
    main()