import cv2
import numpy as np


PANEL_WIDTH = 300


def create_panel(height, counts, fps=0, active_objects=0):
    panel_height = height
    panel_width = PANEL_WIDTH
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    
    y = 20
    cv2.putText(panel, "CONTEO DE PAQUETES", (15, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    y += 40
    
    cv2.line(panel, (10, y-10), (PANEL_WIDTH-10, y-10), (100, 100, 100), 1)
    y += 20
    
    cv2.putText(panel, "Entrantes:", (20, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(panel, str(counts['in']), (180, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    y += 40
    
    cv2.putText(panel, "Salientes:", (20, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 1)
    cv2.putText(panel, str(counts['out']), (180, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
    y += 40
    
    cv2.line(panel, (10, y), (PANEL_WIDTH-10, y), (100, 100, 100), 1)
    y += 20
    
    cv2.putText(panel, "Total:", (20, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(panel, str(counts['total']), (150, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    y += 50
    
    cv2.line(panel, (10, y-10), (PANEL_WIDTH-10, y-10), (100, 100, 100), 1)
    y += 20
    
    cv2.putText(panel, "Objetos activos:", (20, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(panel, str(active_objects), (180, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    y += 40
    
    cv2.putText(panel, "FPS:", (20, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    cv2.putText(panel, str(int(fps)), (100, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    y += 50
    
    cv2.line(panel, (10, y-10), (PANEL_WIDTH-10, y-10), (100, 100, 100), 1)
    y += 20
    
    cv2.putText(panel, "Controles:", (20, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    y += 25
    cv2.putText(panel, "Q - Salir", (20, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 20
    cv2.putText(panel, "R - Resetear conteo", (20, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 20
    cv2.putText(panel, "L - Ajustar linea", (20, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return panel


def combine_frames(main_frame, panel):
    h, w = main_frame.shape[:2]
    panel_h, panel_w = panel.shape[:2]
    
    if h != panel_h:
        panel = cv2.resize(panel, (panel_w, h))
    
    return np.hstack([main_frame, panel])


def draw_counting_line(frame, line_y, direction='down'):
    if line_y is not None:
        color = (0, 255, 255) if direction == 'down' else (255, 100, 0)
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), color, 2)
        
        label = "Linea de conteo" if direction == 'down' else "Linea salida"
        cv2.putText(frame, label, (10, line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame


def create_welcome_screen(width, height):
    screen = np.zeros((height, width, 3), dtype=np.uint8)
    screen[:] = (20, 20, 20)
    
    cv2.putText(screen, "POC Contabilidad de Paquetes", 
               (width//2 - 200, height//2 - 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    cv2.putText(screen, "Presione cualquier tecla para comenzar", 
               (width//2 - 170, height//2 + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    return screen


def create_camera_selection(cameras, width, height):
    screen = np.zeros((height, width, 3), dtype=np.uint8)
    screen[:] = (20, 20, 20)
    
    cv2.putText(screen, "Seleccione Camara", 
               (width//2 - 120, height//2 - 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    y = height//2 - 20
    for i, cam_idx in enumerate(cameras):
        cv2.putText(screen, f"{i+1}. Camara {cam_idx}", 
                   (width//2 - 60, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        y += 30
    
    cv2.putText(screen, "0. Salir", 
               (width//2 - 40, y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    return screen