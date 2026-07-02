import json
import cv2
import mediapipe as mp
import time
import numpy as np
import threading
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from collections import deque


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "modelo_vgg_fatiga.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")
RUTA_ENTRADA = r"C:\Users\scani\Videos\Grabaciones de pantalla\vp8.mp4"


try:
    detector_ia = load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)["class_names"]
    IDX_OJO_ABIERTO = labels.index("abierto")
    print(f"--- IA Cargada exitosamente desde: {MODEL_PATH} ---")
except Exception as e:
    print(f"Error cargando IA o archivos de configuración: {e}")
    raise SystemExit(1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5)

EYE_LEFT = [362, 385, 387, 263, 373, 380]
EYE_RIGHT = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308]

frame_actual = None
resultado_ia = {
    "riesgo": "NORMAL", 
    "color": (0, 255, 0), 
    "ms": 0, 
    "b": 0, 
    "img_L": None, "img_R": None,
    "cat_L": "", "p_L": 0.0, "cat_R": "", "p_R": 0.0,
    "ear_L": 0.0, "ear_R": 0.0
}
running = True

def get_ear(landmarks, eye_points):
    p = [landmarks[i] for i in eye_points]
    ver_dist = abs(p[1].y - p[5].y) + abs(p[2].y - p[4].y)
    hor_dist = abs(p[0].x - p[3].x)
    return ver_dist / (2.0 * hor_dist + 1e-6)

def get_mar(landmarks, mouth_points):
    p = [landmarks[i] for i in mouth_points]
    ver_dist = abs(p[0].y - p[1].y) 
    hor_dist = abs(p[2].x - p[3].x)
    return ver_dist / (hor_dist + 1e-6)

def validar_ojo_ia(frame, lms, idxs, w, h):
    xs = [lms[i].x for i in idxs]; ys = [lms[i].y for i in idxs]
    cx, cy = int(sum(xs)/len(xs)*w), int(sum(ys)/len(ys)*h)
    ancho_ojo_px = abs(lms[idxs[0]].x - lms[idxs[3]].x) * w
    margen = int(ancho_ojo_px * 0.8) 
    x1, y1 = max(0, cx - margen), max(0, cy - margen)
    x2, y2 = min(w, cx + margen), min(h, cy + margen)
    roi = frame[y1:y2, x1:x2]
    
    if roi.size == 0: return False, 0.0, None, "N/A"
    
    roi_ia = cv2.resize(roi, (224, 224))
    img_array = np.array(roi_ia, dtype="float32")
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = preprocess_input(img_array) 

    preds = detector_ia.predict(img_array, verbose=0)[0]
    idx_max = np.argmax(preds)
    return (idx_max == IDX_OJO_ABIERTO), preds[idx_max], cv2.resize(roi, (100, 100)), labels[idx_max].upper()

def hilo_ia():
    global frame_actual, resultado_ia, running
    
    bostezos_hist = deque(); microsuenos_hist = deque()
    inicio_cierre = None; inicio_boca = None
    ya_contado_ms = False; ya_contado_bostezo = False
    
    UMBRAL_EAR = 0.16 
    UMBRAL_MAR = 0.55
    UMBRAL_CONF_IA = 0.75 

    while running:
        if frame_actual is None:
            time.sleep(0.01)
            continue

        img_analizar = frame_actual.copy()
        h, w, _ = img_analizar.shape
        ahora = time.time()
        
        img_rgb = cv2.cvtColor(cv2.resize(img_analizar, (640, 360)), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        while bostezos_hist and ahora - bostezos_hist[0] > 900: bostezos_hist.popleft()
        while microsuenos_hist and ahora - microsuenos_hist[0] > 900: microsuenos_hist.popleft()

        local_res = {
            "riesgo": "NORMAL", "color": (0, 255, 0), 
            "ms": len(microsuenos_hist), "b": len(bostezos_hist),
            "img_L": None, "img_R": None, "cat_L": "", "p_L": 0.0, 
            "cat_R": "", "p_R": 0.0, "ear_L": 0.0, "ear_R": 0.0
        }

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            
            abierto_L, p_L, img_L, cat_L = validar_ojo_ia(img_analizar, lms, EYE_LEFT, w, h)
            abierto_R, p_R, img_R, cat_R = validar_ojo_ia(img_analizar, lms, EYE_RIGHT, w, h)
            ear_L = get_ear(lms, EYE_LEFT); ear_R = get_ear(lms, EYE_RIGHT)
            ear = (ear_L + ear_R) / 2.0; mar = get_mar(lms, MOUTH)

            ojos_cerrados = (ear < UMBRAL_EAR) or (cat_L == "CERRADO" and p_L > UMBRAL_CONF_IA)
            
            if ojos_cerrados:
                if inicio_cierre is None: inicio_cierre = ahora
                if (ahora - inicio_cierre) > 1.5 and not ya_contado_ms:
                    microsuenos_hist.append(ahora); ya_contado_ms = True
            else:
                inicio_cierre = None; ya_contado_ms = False

            if mar > UMBRAL_MAR:
                if inicio_boca is None: inicio_boca = ahora
                if (ahora - inicio_boca) > 1.5 and not ya_contado_bostezo:
                    bostezos_hist.append(ahora); ya_contado_bostezo = True
            else:
                inicio_boca = None; ya_contado_bostezo = False

            c_ms = len(microsuenos_hist); c_b = len(bostezos_hist)
            if c_ms >= 3 or (inicio_cierre and (ahora - inicio_cierre) >= 3.0):
                local_res["riesgo"] = "SEVERO"; local_res["color"] = (0, 0, 255)
            elif c_b >= 4 or (c_ms >= 1 and c_b >= 2):
                local_res["riesgo"] = "MODERADO"; local_res["color"] = (0, 165, 255)
            elif c_b >= 1 or c_ms >= 1:
                local_res["riesgo"] = "LEVE"; local_res["color"] = (0, 255, 255)

            local_res.update({
                "ms": c_ms, "b": c_b, "img_L": img_L, "img_R": img_R,
                "cat_L": cat_L, "p_L": p_L, "cat_R": cat_R, "p_R": p_R,
                "ear_L": ear_L, "ear_R": ear_R
            })

        resultado_ia = local_res

# --- HILO PRINCIPAL ---
def analizar():
    global frame_actual, resultado_ia, running
    cap = cv2.VideoCapture(RUTA_ENTRADA)
    if not cap.isOpened(): print("Error abriendo video"); return
    
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps_original) if fps_original > 0 else 1

    thread_ia = threading.Thread(target=hilo_ia, daemon=True)
    thread_ia.start()

    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.resize(frame, (1080, 600))
        frame_actual = frame.copy() 
        
        res = resultado_ia 
        h, w, _ = frame.shape

        # --- DIBUJAR INTERFAZ ---
        if res["img_L"] is not None:
            frame[20:120, 20:120] = res["img_L"]
            cv2.putText(frame, f"{res['cat_L']} {res['p_L']:.2f}", (20, 135), 1, 1, (255,255,255), 1)
            cv2.putText(frame, f"EAR: {res['ear_L']:.2f}", (20, 155), 1, 1, (0, 255, 255), 1)
            
        if res["img_R"] is not None:
            frame[20:120, 140:240] = res["img_R"]
            cv2.putText(frame, f"{res['cat_R']} {res['p_R']:.2f}", (140, 135), 1, 1, (255,255,255), 1)
            cv2.putText(frame, f"EAR: {res['ear_R']:.2f}", (140, 155), 1, 1, (0, 255, 255), 1)

        cv2.rectangle(frame, (0, h-80), (w, h), (0,0,0), -1)
        cv2.putText(frame, f"RIESGO: {res['riesgo']}", (30, h-30), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, res['color'], 2)
        
        info_contadores = f"MS: {res['ms']}   BOSTEZOS: {res['b']}"
        cv2.putText(frame, info_contadores, (w-450, h-35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Analizador de Fatiga - TIEMPO REAL", frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    analizar()