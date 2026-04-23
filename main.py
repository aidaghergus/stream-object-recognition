import csv
import cv2
from datetime import datetime
from ultralytics import YOLO
from vidgear.gears import CamGear

# 1. Încărcăm modelul pre-antrenat
model = YOLO('yolov8n.pt')

# 2. Definim link-ul de YouTube (poate fi Live Stream sau video normal)
# Exemplu: Live stream din New York sau orice alt video de pe YouTube
youtube_url = 'https://www.youtube.com/watch?v=1EiC9bvVGnk' 

# Inițializăm accesul la stream-ul de YouTube folosind CamGear
print("Conectare la YouTube în curs...")
stream = CamGear(source=youtube_url, stream_mode=True, logging=True).start()

# ==========================================
# CSV Logging Setup
# ==========================================
CSV_FILE = "raport_trafic.csv"
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timp", "Masini", "Pietoni"])

frame_count = 0
LOG_INTERVAL = 30  # salvează o dată pe secundă (~30fps)

print("Procesare începută... Apasă tasta 'q' pe tastatură pentru a opri.")

while True:
    # Citim cadrul de la stream-ul de YouTube
    frame = stream.read()

    # Dacă frame-ul este None, înseamnă că video-ul s-a terminat sau a picat netul
    if frame is None:
        print("Stream-ul s-a terminat sau s-a întrerupt.")
        break

    frame_count += 1

    # 3. Rulăm modelul de detecție
    # Detectăm doar Persoane(0), Mașini(2) și Motociclete(3)
    results = model(frame, classes=[0, 2, 3], conf=0.4, verbose=False)

    # 4. Numărăm obiectele pe categorii
    numar_masini = 0
    numar_pietoni = 0
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls == 0:
                numar_pietoni += 1
            elif cls in [2, 3]:  # mașini + motociclete
                numar_masini += 1

    # 5. Desenăm rezultatele pe imagine
    annotated_frame = results[0].plot()

    # 6. HUD cu statistici
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)

    cv2.putText(annotated_frame, f"Masini:  {numar_masini}", (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
    cv2.putText(annotated_frame, f"Pietoni: {numar_pietoni}", (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)

    # Salvare CSV la fiecare LOG_INTERVAL cadre
    if frame_count % LOG_INTERVAL == 0:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, numar_masini, numar_pietoni])

    # 7. Afișăm cadrul procesat
    cv2.imshow("YouTube Live Object Detection", annotated_frame)

    # Apasă 'q' pentru a închide fereastra în siguranță
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Eliberăm resursele la final
stream.stop()
cv2.destroyAllWindows()