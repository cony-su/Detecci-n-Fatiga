import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

# Configuración
ruta_val = r"C:\Users\scani\vgg\VGGPractica\dataset\val"
IMG_SIZE = (224, 224)
modelos = [f for f in os.listdir('.') if f.endswith('.h5')]

def evaluar_modelo_integral(nombre_modelo):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    gen = datagen.flow_from_directory(
        ruta_val, 
        target_size=IMG_SIZE, 
        batch_size=32, 
        class_mode='categorical', 
        shuffle=False
    )
    
    model = tf.keras.models.load_model(nombre_modelo)
    y_true = gen.classes
    preds = model.predict(gen, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    
    reporte = classification_report(y_true, y_pred, target_names=list(gen.class_indices.keys()), output_dict=True)
    
    f1_global = reporte['accuracy'] # Accuracy total
    recall_cerrado = reporte['cerrado']['recall']
    f1_cerrado = reporte['cerrado']['f1-score']
    
    cm = confusion_matrix(y_true, y_pred)
    idx_cerrado = gen.class_indices['cerrado']
    falsos_negativos = sum(cm[idx_cerrado]) - cm[idx_cerrado][idx_cerrado]

    return {
        "F1_Global": f1_global,
        "Recall_Cerrado": recall_cerrado,
        "F1_Cerrado": f1_cerrado,
        "Falsos_Negativos": falsos_negativos
    }

print(f"{'MODELO':<25} | {'F1 GLOBAL':<10} | {'RECALL CERR.':<12} | {'ERRORES CRÍTICOS'}")
print("-" * 75)

tabla_comparativa = []

for nombre in modelos:
    try:
        m = evaluar_modelo_integral(nombre)
        print(f"{nombre:<25} | {m['F1_Global']:.4f}    | {m['Recall_Cerrado']:.4f}     | {int(m['Falsos_Negativos'])} fotos")
        
        m['Nombre'] = nombre
        tabla_comparativa.append(m)
    except Exception as e:
        print(f"Error con {nombre}: {e}")

# --- CRITERIO DE SELECCIÓN ---
if tabla_comparativa:
    tabla_comparativa.sort(key=lambda x: (x['Recall_Cerrado'], x['F1_Global']), reverse=True)
    
    print("-" * 75)
    ganador = tabla_comparativa[0]
    print(f"RECOMENDACIÓN TÉCNICA: {ganador['Nombre']}")
    print(f"Este modelo detecta el {ganador['Recall_Cerrado']*100:.1f}% de los ojos cerrados.")