import cv2
import numpy as np


def get_available_cameras(max_cameras=10):
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available


def get_camera_info(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    return {
        'index': camera_index,
        'width': width,
        'height': height,
        'fps': fps
    }


def select_camera():
    cameras = get_available_cameras()
    
    if not cameras:
        print("No se encontraron cámaras disponibles")
        return None
    
    print("\n=== Cámaras Disponibles ===")
    for i, cam_idx in enumerate(cameras):
        info = get_camera_info(cam_idx)
        if info:
            print(f"{i + 1}. Cámara {cam_idx} - {info['width']}x{info['height']} @ {info['fps']:.1f} fps")
        else:
            print(f"{i + 1}. Cámara {cam_idx}")
    
    if len(cameras) == 1:
        print(f"\nSolo se encontró una cámara ({cameras[0]}), seleccionándola automáticamente...")
        return cameras[0]
    
    print("\nSeleccione el número de cámara (0 para salir): ", end="")
    try:
        selection = int(input())
        if selection == 0:
            return None
        if 1 <= selection <= len(cameras):
            return cameras[selection - 1]
    except ValueError:
        pass
    
    return cameras[0]