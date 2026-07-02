import cv2
import mediapipe as mp
import os

INPUT_DIR = r"C:\Users\scani\Downloads\pivote"  
OUTPUT_DIR = r"C:\Users\scani\Downloads\resultadoPivote"
IMG_SIZE = 224

LEFT = [33, 133]
RIGHT = [362, 263]

# =======================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)

def crop_eye(img, p1, p2, pad=20):
    h, w, _ = img.shape
    x1 = int(min(p1.x, p2.x) * w) - pad
    y1 = int(min(p1.y, p2.y) * h) - pad
    x2 = int(max(p1.x, p2.x) * w) + pad
    y2 = int(max(p1.y, p2.y) * h) + pad

    return img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]

count = 0

for file in os.listdir(INPUT_DIR):
    path = os.path.join(INPUT_DIR, file)

    img = cv2.imread(path)
    if img is None:
        continue

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = face.process(rgb)

    if not res.multi_face_landmarks:
        continue

    lm = res.multi_face_landmarks[0].landmark

    left_eye = crop_eye(img, lm[LEFT[0]], lm[LEFT[1]])
    right_eye = crop_eye(img, lm[RIGHT[0]], lm[RIGHT[1]])

    if left_eye.size > 0:
        left_eye = cv2.resize(left_eye, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"eye_{count:05}_L.jpg"), left_eye)

    if right_eye.size > 0:
        right_eye = cv2.resize(right_eye, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"eye_{count:05}_R.jpg"), right_eye)

    count += 1

print(f"Listo. Procesadas {count} caras.")
print("Ojos guardados en:", OUTPUT_DIR)
