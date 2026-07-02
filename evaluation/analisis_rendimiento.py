import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# 1. Cargar el modelo guardado
model = load_model('modelo_vgg_fatiga.h5')

val_datagen = ImageDataGenerator(rescale=1./255)
val_gen = val_datagen.flow_from_directory(
    r"C:\Users\scani\vgg\VGGPractica\dataset\val",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False 
)

# 3. Obtener predicciones
y_pred = model.predict(val_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_gen.classes
class_names = list(val_gen.class_indices.keys())

# 4. Generar Matriz de Confusión
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicción (Modelo)')
plt.ylabel('Realidad (Etiqueta)')
plt.title('Matriz de Confusión V3 - Proyecto Fatiga')
plt.show()

# 5. Mostrar F1-Score y métricas detalladas
print(classification_report(y_true, y_pred_classes, target_names=class_names))