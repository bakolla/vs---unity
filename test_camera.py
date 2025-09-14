import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Kamera działa pod indeksem {i}")
        cap.release()
    else:
        print(f"Brak kamery pod indeksem {i}")
