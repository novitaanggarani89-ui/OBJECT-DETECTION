import cv2
from ultralytics import YOLO

# Memuat model YOLOv8 versi ringan (nano)
model = YOLO("yolov8n.pt")

# ID kelas COCO dataset: 0 = person, 67 = cell phone
TARGET_CLASSES = [0, 67]

# Membuka kamera webcam (0 adalah indeks kamera default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Kamera tidak dapat diakses.")
    exit()

print("Deteksi berjalan... Tekan 'q' pada jendela video untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Jalankan deteksi hanya untuk kelas orang dan HP
    results = model(frame, classes=TARGET_CLASSES, verbose=False)

    # Gambar bounding box dan label pada frame
    annotated_frame = results[0].plot()

    # Tampilkan hasil di jendela video
    cv2.imshow("Deteksi Real-Time: Orang & HP", annotated_frame)

    # Tekan tombol 'q' di keyboard untuk menghentikan program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()
