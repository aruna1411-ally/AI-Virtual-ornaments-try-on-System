import cv2
import dlib
import os
import numpy as np

# =============================
# PATHS
# =============================
BASE = os.path.dirname(os.path.abspath(__file__))
EARRING_DIR = os.path.join(BASE, "ornaments", "earring")
NECKLACE_DIR = os.path.join(BASE, "ornaments", "necklace")
PREDICTOR_PATH = os.path.join(BASE, "models", "shape_predictor_68_face_landmarks.dat")

# =============================
# USER CHOICE
# =============================
choice = input("Choose ornament (earring / necklace / both): ").strip().lower()

# =============================
# LOAD MODEL
# =============================
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# =============================
# LOAD PNGS
# =============================
def load_pngs(folder):
    imgs = []
    for f in os.listdir(folder):
        if f.lower().endswith(".png"):
            path = os.path.join(folder, f)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] == 4:
                imgs.append(img)
    return imgs


earrings = load_pngs(EARRING_DIR)
necklaces = load_pngs(NECKLACE_DIR)

print("Loaded Earrings:", len(earrings))
print("Loaded Necklaces:", len(necklaces))

ear_index = 0
neck_index = 0

# =============================
# OVERLAY
# =============================
def overlay(frame, img, x, y):
    h, w = img.shape[:2]

    # clip safely
    if x < 0:
        x = 0
    if y < 0:
        y = 0

    if x + w > frame.shape[1]:
        w = frame.shape[1] - x
        img = img[:, :w]

    if y + h > frame.shape[0]:
        h = frame.shape[0] - y
        img = img[:h]

    if h <= 0 or w <= 0:
        return

    alpha = img[:, :, 3] / 255.0
    alpha_inv = 1 - alpha

    for c in range(3):
        frame[y:y+h, x:x+w, c] = (
            alpha * img[:, :, c] +
            alpha_inv * frame[y:y+h, x:x+w, c]
        )

# =============================
# CAMERA
# =============================
cap = cv2.VideoCapture(0)

print("Click webcam window first")
print("E/Q → Next/Prev Earrings")
print("N/B → Next/Prev Necklace")
print("ESC → Exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)

        # ---------------- EARRING ----------------
        if choice in ["earring", "both"] and len(earrings) > 0:
            img = earrings[ear_index]

            face_h = face.bottom() - face.top()
            ear_h = int(face_h * 0.30)
            ear_w = int(ear_h * img.shape[1] / img.shape[0])

            earring = cv2.resize(img, (ear_w, ear_h))

            lx = shape.part(2).x
            ly = shape.part(2).y
            rx = shape.part(14).x
            ry = shape.part(14).y

            overlay(frame, earring, lx - ear_w // 2, ly)
            overlay(frame, earring, rx - ear_w // 2, ry)

        # ---------------- NECKLACE ----------------
        if choice in ["necklace", "both"] and len(necklaces) > 0:
            img = necklaces[neck_index]

            left = shape.part(4)
            right = shape.part(12)

            neck_width = int(abs(right.x - left.x) * 1.2)
            neck_h = int(neck_width * img.shape[0] / img.shape[1])

            necklace = cv2.resize(img, (neck_width, neck_h))

            x = left.x - int(neck_width * 0.1)
            y = shape.part(8).y + 5

            overlay(frame, necklace, x, y)

    # show indexes
    cv2.putText(frame, f"Earring: {ear_index+1}/{len(earrings)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(frame, f"Necklace: {neck_index+1}/{len(necklaces)}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("AI Ornament Try-On", frame)

    key = cv2.waitKey(30) & 0xFF

    if key == ord("e") and len(earrings) > 0:
        ear_index = (ear_index + 1) % len(earrings)

    elif key == ord("q") and len(earrings) > 0:
        ear_index = (ear_index - 1) % len(earrings)

    elif key == ord("n") and len(necklaces) > 0:
        neck_index = (neck_index + 1) % len(necklaces)

    elif key == ord("b") and len(necklaces) > 0:
        neck_index = (neck_index - 1) % len(necklaces)

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
