import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model

# configuracion
ruta_dataset = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/dataset_secuencias_balanceado.jsonl"
ruta_modelo = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/modelo_transformer_f.keras"
longitud_max_mirna = 30
longitud_max_transcrito = 300

# codificar secuencias 
def codificar_secuencia(seq, maxlen):
    mapa = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
    codificada = np.zeros((maxlen, 4), dtype=np.float32)
    for i, base in enumerate(seq.upper()[:maxlen]):
        if base in mapa:
            codificada[i, mapa[base]] = 1.0
    return codificada

# carga de datos
X_mirna, X_transcrito, y = [], [], []
with open(ruta_dataset, "r", encoding="utf-8") as f:
    for linea in f:
        d = json.loads(linea)
        X_mirna.append(codificar_secuencia(d["secuencia_mirna"], longitud_max_mirna))
        X_transcrito.append(codificar_secuencia(d["secuencia_transcrito"], longitud_max_transcrito))
        y.append(d["etiqueta"])

X_mirna = np.array(X_mirna)
X_transcrito = np.array(X_transcrito)
y = np.array(y)

# separar el conjunto de validacion 
_, X_mirna_val, _, X_trans_val, _, y_val = train_test_split(
    X_mirna, X_transcrito, y, test_size=0.2, random_state=42, stratify=y
)

# cargar el modelo
modelo = load_model(ruta_modelo)

# Predecir
probabilidades = modelo.predict([X_mirna_val, X_trans_val])
predicciones = (probabilidades > 0.5).astype(int).flatten()

#  metricas

print(classification_report(y_val, predicciones, target_names=["NEGATIVO", "POSITIVO"]))

#matriz de confusion
cm = confusion_matrix(y_val, predicciones)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap='Blues')
plt.title("Matriz de Confusi\u00f3n")
plt.colorbar()
plt.xticks([0, 1], ["NEGATIVO", "POSITIVO"])
plt.yticks([0, 1], ["NEGATIVO", "POSITIVO"])
plt.xlabel("Predicci\u00f3n")
plt.ylabel("Real")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.tight_layout()
plt.show()

# curva roc
fpr, tpr, _ = roc_curve(y_val, probabilidades)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("Tasa de falsos positivos")
plt.ylabel("Tasa de verdaderos positivos")
plt.title("Curva ROC")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
