import json
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input  
import cv2
import mediapipe as mp
import time
import numpy as np
import platform
import math
from collections import deque
import os
import csv
from datetime import datetime

def registrar_evento(tipo_evento, frame, nivel_riesgo, ear_val, mar_val):
    path_fotos = "eventos/fotos"
    if not os.path.exists(path_fotos): os.makedirs(path_fotos)
    ahora_dt = datetime.now()
    evento_id = ahora_dt.strftime("%Y%m%d_%H%M%S_%f")
    nombre_foto = f"EVD_{evento_id}.jpg"

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ancho_estandar = 480 
    alto = int(frame.shape[0] * (ancho_estandar / frame.shape[1]))
    frame_reporte = cv2.resize(gris, (ancho_estandar, alto), interpolation=cv2.INTER_AREA)
    frame_reporte = cv2.equalizeHist(frame_reporte)

    ruta_completa = os.path.join(path_fotos, nombre_foto)
    cv2.imwrite(ruta_completa, frame_reporte, [cv2.IMWRITE_JPEG_QUALITY, 20])

    reporte_path = "eventos/reporte_fatiga.csv"
    archivo_nuevo = not os.path.isfile(reporte_path)
    with open(reporte_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if archivo_nuevo:
            writer.writerow(["ID_EVENTO", "Fecha", "Tipo", "Riesgo", "EAR", "MAR", "Foto"])
        writer.writerow([evento_id, ahora_dt.strftime("%Y-%m-%d %H:%M:%S"), tipo_evento, nivel_riesgo, f"{ear_val:.4f}", f"{mar_val:.4f}", nombre_foto])

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

MODEL_NAME = os.path.join('models', 'modelo_vgg_fatiga.h5')

try:
    detector_ia = load_model(MODEL_NAME)
    with open("labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)["class_names"]
    
    IDX_ABIERTO = labels.index("abierto")
    IDX_CERRADO = labels.index("cerrado")
    IDX_OBSTRUIDO = labels.index("obstruido")
    
except Exception as e:
    print(f"Error cargando IA: {e}")
    raise SystemExit(1)

def validar_ojo_ia(frame, lms, idxs, w, h):
    p_interno = lms[idxs[0]]
    p_externo = lms[idxs[3]]
    
    xs = [lms[i].x for i in idxs]
    ys = [lms[i].y for i in idxs]
    cx, cy = int(sum(xs)/len(xs)*w), int(sum(ys)/len(ys)*h)
    
    ancho_ojo_px = abs(p_interno.x - p_externo.x) * w
    margen = int(ancho_ojo_px * 0.8) 
    
    x1, y1 = max(0, cx - margen), max(0, cy - margen)
    x2, y2 = min(w, cx + margen), min(h, cy + margen)
    
    roi = frame[y1:y2, x1:x2]
    
    if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
        return "desconocido", 0.0, None, "N/A"

    roi_ia = cv2.resize(roi, (224, 224))
    
    img_array = roi_ia.astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  
    
    preds = detector_ia.predict(img_array, verbose=0)[0]
    
    idx_max = np.argmax(preds)
    nombre_categoria = labels[idx_max].upper()
    prob_max = preds[idx_max]
    
    if idx_max == IDX_ABIERTO:
        estado = "abierto"
    elif idx_max == IDX_CERRADO:
        estado = "cerrado"
    elif idx_max == IDX_OBSTRUIDO:
        estado = "obstruido"
    else:
        estado = "desconocido"
    
    roi_mini = cv2.resize(roi, (100, 100))
    
    return estado, prob_max, roi_mini, nombre_categoria

def play_beep(freq, dur):
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(freq, dur)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.6)

EYE_LEFT = [362, 385, 387, 263, 373, 380]
EYE_RIGHT = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308]

UMBRAL_EAR = 0.18
UMBRAL_MAR = 0.55
UMBRAL_CONF_IA = 0.30  
TIEMPO_BOSTEZO_MIN = 1.5
VENTANA_TIEMPO = 900

bostezos_hist = deque()
microsuenos_hist = deque()
inicio_cierre = None
inicio_boca = None
ya_contado_ms = False
ya_contado_bostezo = False

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    ahora = time.time()
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    while bostezos_hist and ahora - bostezos_hist[0] > VENTANA_TIEMPO:
        bostezos_hist.popleft()
    while microsuenos_hist and ahora - microsuenos_hist[0] > VENTANA_TIEMPO:
        microsuenos_hist.popleft()

    estado_riesgo = "NORMAL"
    color_riesgo = (0, 255, 0)
    mar, ear = 0.0, 0.0

    if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0].landmark
        
        estado_L, p_L, img_L, cat_L = validar_ojo_ia(frame, lms, EYE_LEFT, w, h)
        estado_R, p_R, img_R, cat_R = validar_ojo_ia(frame, lms, EYE_RIGHT, w, h)
        
        ear_L = get_ear(lms, EYE_LEFT)
        ear_R = get_ear(lms, EYE_RIGHT)
        ear = (ear_L + ear_R) / 2.0

        def get_color_txt(estado, prob):
            """Color según estado y confianza"""
            if prob < UMBRAL_CONF_IA:
                return (0, 0, 255)  
            elif estado == "abierto":
                return (0, 255, 0)  
            elif estado == "cerrado":
                return (0, 165, 255)  
            elif estado == "obstruido":
                return (255, 0, 0)  
            else:
                return (255, 255, 255)  

        if img_L is not None:
            frame[20:120, 20:120] = img_L
            cv2.rectangle(frame, (20, 120), (140, 175), (255, 255, 255), -1)
            cv2.putText(frame, f"IA: {estado_L.upper()}", (22, 135), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, get_color_txt(estado_L, p_L), 1)
            cv2.putText(frame, f"P: {p_L:.2f}", (22, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
            cv2.putText(frame, f"EAR: {ear_L:.3f}", (22, 165), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

        if img_R is not None:
            frame[180:280, 20:120] = img_R
            cv2.rectangle(frame, (20, 280), (140, 335), (255, 255, 255), -1)
            cv2.putText(frame, f"IA: {estado_R.upper()}", (22, 295), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, get_color_txt(estado_R, p_R), 1)
            cv2.putText(frame, f"P: {p_R:.2f}", (22, 310), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
            cv2.putText(frame, f"EAR: {ear_R:.3f}", (22, 325), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

        ojo_izq_cerrado = (ear_L < UMBRAL_EAR) or (estado_L == "cerrado" and p_L > UMBRAL_CONF_IA)
        ojo_der_cerrado = (ear_R < UMBRAL_EAR) or (estado_R == "cerrado" and p_R > UMBRAL_CONF_IA)
        
        ojos_cerrados_count = sum([ojo_izq_cerrado, ojo_der_cerrado])
        
        if ojos_cerrados_count >= 1:
            if inicio_cierre is None:
                inicio_cierre = ahora
            if (ahora - inicio_cierre) > 1.5:
                if not ya_contado_ms:
                    microsuenos_hist.append(ahora)
                    registrar_evento("MICROSUEÑO", frame, "ALTO", ear, mar)
                    ya_contado_ms = True
                play_beep(1000, 100)
        else:
            inicio_cierre = None
            ya_contado_ms = False

        mar = get_mar(lms, MOUTH)
        if mar > UMBRAL_MAR:
            if inicio_boca is None:
                inicio_boca = ahora
            if (ahora - inicio_boca) > TIEMPO_BOSTEZO_MIN:
                if not ya_contado_bostezo:
                    bostezos_hist.append(ahora)
                    registrar_evento("BOSTEZO", frame, "N/A", ear, mar)
                    ya_contado_bostezo = True
        else:
            inicio_boca = None
            ya_contado_bostezo = False

        c_ms = len(microsuenos_hist)
        c_b = len(bostezos_hist)
        
        if c_ms >= 3 or (c_ms >= 1 and c_b >= 4) or (inicio_cierre and (ahora - inicio_cierre) > 2.0):
            estado_riesgo = "SEVERO"
            color_riesgo = (0, 0, 255)
        elif c_b >= 2 or c_ms >= 1:
            estado_riesgo = "MODERADO"
            color_riesgo = (0, 165, 255)
        else:
            estado_riesgo = "NORMAL"
            color_riesgo = (0, 255, 0)

    cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"RIESGO: {estado_riesgo}", (20, h - 15), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, color_riesgo, 2)
    cv2.putText(frame, f"MS: {len(microsuenos_hist)} | B: {len(bostezos_hist)}", 
                (w - 220, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Driver Monitor Fatiga", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()