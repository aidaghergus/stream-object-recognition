# Vrei și mai multă viteză? (Pentru nivelul de Producție)
# Dacă vreodată vrei să transformi acest script într-o aplicație reală vândută unui club de fotbal, scripturile .pt (PyTorch) nu se folosesc live. Modelul trebuie "compilat" pentru hardware-ul tău specific.

# Dacă ai o placă video NVIDIA, deschide un terminal și rulează:
# Acest proces va dura vreo 5-10 minute și va crea un fișier numit yolov8m.engine (TensorRT). Dacă folosești acest fișier în cod în loc de yolov8m.pt, viteza aplicației tale va crește cu încă 300%!
#yolo export model=yolov8m.pt format=engine half=True


import csv
import cv2
import numpy as np
import torch
from datetime import datetime
from ultralytics import YOLO
from vidgear.gears import CamGear
from sklearn.cluster import KMeans
from collections import Counter, defaultdict

# ==========================================
# 1. Hardware și Model (Folosim versiunea Large)
# ==========================================
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# YOLOv8 Large (mai precis, recunoaște oameni chiar și din bucăți de corp)
model = YOLO('yolov8m.pt') #YOLO('yolov8l.pt') 
model.to(device)

youtube_url = 'https://www.youtube.com/watch?v=5mTGRWX1Vcg'
stream = CamGear(source=youtube_url, stream_mode=True, logging=False).start()

# ==========================================
# 2. Extragerea culorilor în spațiul HSV (Imun la Umbre)
# ==========================================
def extract_jersey_color_hsv(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1
    y_start, y_end = y1 + int(h * 0.15), y1 + int(h * 0.40)
    x_start, x_end = x1 + int(w * 0.25), x2 - int(w * 0.25)
    
    if y_start >= y_end or x_start >= x_end or y_start < 0 or x_start < 0:
        return None
        
    roi = frame[y_start:y_end, x_start:x_end]
    if roi.size == 0: return None
    
    # CONVERSIA MAGICĂ: BGR -> HSV
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Calculăm media nuanței, saturației și luminozității
    return np.mean(hsv_roi, axis=(0, 1))

# ==========================================
# 3. Memorie și Variabile
# ==========================================
kmeans = None
is_kmeans_trained = False
colors_for_training = []
team_labels = [] 

player_history = defaultdict(list)

team_colors_draw = {
    0: (0, 0, 255),    # Echipa 1
    1: (255, 0, 0),    # Echipa 2
    -1: (150, 150, 150) # Altul
}

# ==========================================
# CSV Logging Setup
# ==========================================
CSV_FILE = "raport_fotbal.csv"
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timp", "Numar_Jucatori"])

frame_count = 0
LOG_INTERVAL = 30  # salvează o dată pe secundă (~30fps)

print("Scanare tactică avansată... Așteaptă stabilizarea algoritmului HSV.")

while True:
    frame = stream.read()
    if frame is None:
        break

    frame_count += 1

    # conf=0.15 (detectează ușor), iou=0.6 (permite jucători suprapuși), imgsz=1280 (calitate maximă)
    results = model.track(frame,
        classes=[0], 
        persist=True, 
        tracker="bytetrack.yaml", 
                          conf=0.15, 
                          iou=0.5, 
                          imgsz=736, # Rezoluție scăzută de la 1280 la 736
                          half=True, # MAGIC SPEED BOOST (doar dacă ai placă video)
                          verbose=False, 
                          device=device)
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        
        current_frame_colors = []
        valid_indices = []

        for i, box in enumerate(boxes):
            # Folosim noua funcție HSV
            color = extract_jersey_color_hsv(frame, box)
            if color is not None:
                current_frame_colors.append(color)
                valid_indices.append(i)

        # ==========================================
        # 4. Antrenare K-Means pe culori HSV
        # ==========================================
        # if not is_kmeans_trained:
        #     colors_for_training.extend(current_frame_colors)
        #     if len(colors_for_training) > 200: # Am crescut eșantionul pentru o decizie perfectă
        #         print("Extragem echipele folosind spectrul HSV...")
        #         # Folosim 5 clustere (Echipa 1, Echipa 2, Portar 1, Portar 2, Arbitru/Antrenori)
        #         kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
        #         labels = kmeans.fit_predict(colors_for_training)
                
        #         label_counts = Counter(labels)
        #         # Cele mai mari 2 grupuri sunt GARANTAT cele 2 echipe de pe teren
        #         team_labels = [item[0] for item in label_counts.most_common(2)]
        #         is_kmeans_trained = True
        #         print("Sistem stabilizat. Trecem pe mod live.")
        # ==========================================
        # 4. Antrenare K-Means (Optimizat la 3 Clase)
        # ==========================================
        if not is_kmeans_trained:
            colors_for_training.extend(current_frame_colors)
            
            # 150-200 de detecții sunt suficiente pentru a stabili culorile
            if len(colors_for_training) > 150: 
                print("Extragem cele 2 echipe principale + 1 clasă pentru arbitri/portari...")
                
                # Modificare critică: n_clusters=3 (Echipa 1, Echipa 2, Alții)
                kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
                labels = kmeans.fit_predict(colors_for_training)
                
                label_counts = Counter(labels)
                
                # Cele mai mari 2 grupuri sunt GARANTAT echipele de pe teren
                # Grupul cu cei mai puțini oameni va fi grupul arbitrului/portarilor
                team_labels = [item[0] for item in label_counts.most_common(2)]
                
                is_kmeans_trained = True
                print("Echipe identificate cu succes! Trecem pe live.")    
        # ==========================================
        # 5. Clasificare cu Memorie Stabilă
        # ==========================================
        else:
            if len(current_frame_colors) > 0:
                instant_labels = kmeans.predict(current_frame_colors)
                
                for j, index in enumerate(valid_indices):
                    x1, y1, x2, y2 = map(int, boxes[index])
                    track_id = int(track_ids[index])
                    instant_cluster = instant_labels[j]
                    
                    player_history[track_id].append(instant_cluster)
                    
                    # Păstrăm o istorie de 45 de cadre (1.5 secunde) pentru o decizie solidă
                    if len(player_history[track_id]) > 45:
                        player_history[track_id].pop(0)
                        
                    stable_cluster = Counter(player_history[track_id]).most_common(1)[0][0]
                    
                    if stable_cluster == team_labels[0]:
                        team_id = 0
                    elif stable_cluster == team_labels[1]:
                        team_id = 1
                    else:
                        team_id = -1
                    
                    draw_color = team_colors_draw[team_id]
                    thickness = 2 if team_id != -1 else 1
                    
                    # Desenăm box-ul
                    cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, thickness)
                    
                    # Facem textul mai curat, cu un fundal negru pentru a fi vizibil pe orice suprafață
                    label_text = f"E.{team_id+1}|{track_id}" if team_id != -1 else f"Ref|{track_id}"
                    
                    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_width, y1), draw_color, -1) # Fundal text
                    cv2.putText(frame, label_text, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ==========================================
    # 6. Numărare și HUD
    # ==========================================
    numar_jucatori = len(results[0].boxes) if results[0].boxes is not None else 0

    # Fundal semi-transparent pentru HUD
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"Jucatori detectati: {numar_jucatori}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Salvare CSV la fiecare LOG_INTERVAL cadre
    if frame_count % LOG_INTERVAL == 0:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, numar_jucatori])

    frame_show = cv2.resize(frame, (1280, 720))
    cv2.imshow("Pro Football Tracker (HSV + YOLOv8l)", frame_show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()