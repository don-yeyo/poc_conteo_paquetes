import numpy as np
import cv2
from collections import deque


class ObjectTracker:
    def __init__(self, max_disappeared=30, iou_threshold=0.1):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
    
    def register(self, bbox):
        self.objects[self.next_id] = {
            'bbox': bbox,
            'trail': deque(maxlen=20)
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        return self.next_id - 1
    
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def compute_iou(self, bbox1, bbox2):
        x1_min, y1_min, w1, h1 = bbox1
        x2_min, y2_min, w2, h2 = bbox2
        
        x1_max = x1_min + w1
        y1_max = y1_min + h1
        x2_max = x2_min + w2
        y2_max = y2_min + h2
        
        xi1 = max(x1_min, x2_min)
        yi1 = max(y1_min, y2_min)
        xi2 = min(x1_max, x2_max)
        yi2 = min(y1_max, y2_max)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        
        union_area = bbox1_area + bbox2_area - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area
    
    def update(self, detections):
        if not detections or not detections.detections:
            for object_id in list(self.objects.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        bboxes = []
        for det in detections.detections:
            bbox = det.bounding_box
            bboxes.append((int(bbox.origin_x), int(bbox.origin_y), 
                         int(bbox.width), int(bbox.height)))
        
        if len(self.objects) == 0:
            for bbox in bboxes:
                self.register(bbox)
            return self.objects
        
        object_ids = list(self.objects.keys())
        used_bboxes = set()
        
        for object_id in object_ids:
            best_iou = 0
            best_bbox_idx = -1
            
            for idx, bbox in enumerate(bboxes):
                if idx in used_bboxes:
                    continue
                
                iou = self.compute_iou(self.objects[object_id]['bbox'], bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_bbox_idx = idx
            
            if best_iou > self.iou_threshold and best_bbox_idx >= 0:
                self.objects[object_id]['bbox'] = bboxes[best_bbox_idx]
                self.objects[object_id]['trail'].append(bboxes[best_bbox_idx])
                self.disappeared[object_id] = 0
                used_bboxes.add(best_bbox_idx)
            else:
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        
        for idx, bbox in enumerate(bboxes):
            if idx not in used_bboxes:
                self.register(bbox)
        
        return self.objects


class PackageCounter:
    def __init__(self, line_y=None, direction='down'):
        self.count_in = 0
        self.count_out = 0
        self.tracked_objects = {}
        self.line_y = line_y
        self.direction = direction
    
    def set_counting_line(self, y, direction='down'):
        self.line_y = y
        self.direction = direction
    
    def check_crossing(self, object_id, prev_bbox, curr_bbox):
        if self.line_y is None:
            return False
        
        prev_center_y = prev_bbox[1] + prev_bbox[3] // 2
        curr_center_y = curr_bbox[1] + curr_bbox[3] // 2
        
        if self.direction == 'down':
            if prev_center_y < self.line_y <= curr_center_y:
                return True, 'in'
            elif prev_center_y > self.line_y >= curr_center_y:
                return True, 'out'
        else:
            if prev_center_y > self.line_y >= curr_center_y:
                return True, 'out'
            elif prev_center_y < self.line_y <= curr_center_y:
                return True, 'in'
        return False, None
    
    def update(self, objects):
        for object_id, obj_data in objects.items():
            curr_bbox = obj_data['bbox']
            trail = obj_data['trail']
            
            if len(trail) >= 2:
                prev_bbox = trail[-2]
                
                if object_id not in self.tracked_objects:
                    self.tracked_objects[object_id] = {'crossed_in': False, 'crossed_out': False}
                
                crossed, direction = self.check_crossing(object_id, prev_bbox, curr_bbox)
                
                if crossed:
                    if direction == 'in' and not self.tracked_objects[object_id]['crossed_in']:
                        self.count_in += 1
                        self.tracked_objects[object_id]['crossed_in'] = True
                    elif direction == 'out' and not self.tracked_objects[object_id]['crossed_out']:
                        self.count_out += 1
                        self.tracked_objects[object_id]['crossed_out'] = True
        
        for object_id in list(self.tracked_objects.keys()):
            if object_id not in objects:
                del self.tracked_objects[object_id]
    
    def get_counts(self):
        return {
            'in': self.count_in,
            'out': self.count_out,
            'total': self.count_in - self.count_out
        }
    
    def reset(self):
        self.count_in = 0
        self.count_out = 0
        self.tracked_objects = {}


def draw_tracked_objects(frame, objects, color=(0, 255, 0), thickness=2):
    for object_id, obj_data in objects.items():
        bbox = obj_data['bbox']
        x, y, w, h = bbox
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        cv2.putText(frame, f"ID:{object_id}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if len(obj_data['trail']) > 1:
            trail = list(obj_data['trail'])
            for i in range(1, len(trail)):
                pt1 = (trail[i-1][0] + trail[i-1][2]//2, trail[i-1][1] + trail[i-1][3]//2)
                pt2 = (trail[i][0] + trail[i][2]//2, trail[i][1] + trail[i][3]//2)
                cv2.line(frame, pt1, pt2, (0, 255, 255), 1)
    
    return frame