import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model

#  Configuración 
ruta_dataset = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/dataset_secuencias_balanceado.jsonl"
rutas_modelos = {
    "Modelo Transformers": "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/modelo_transformer_f.keras",
    "Modelo CNN": "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/modelo_cnn_attention_final.keras"
}
longitud_max_mirna = 30
longitud_max_transcrito = 300

# Función para codificar secuencias 
def codificar_secuencia(seq, maxlen):
    mapa = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
    codificada = np.zeros((maxlen, 4), dtype=np.float32)
    for i, base in enumerate(seq.upper()[:maxlen]):
        if base in mapa:
            codificada[i, mapa[base]] = 1.0
    return codificada

#  Cargar datos 
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

#  Separar validación 
_, X_mirna_val, _, X_trans_val, _, y_val = train_test_split(
    X_mirna, X_transcrito, y, test_size=0.2, random_state=42, stratify=y
)

# Diccionario para almacenar resultados 
resultados = {}

# Evaluar cada modelo 
for nombre_modelo, ruta_modelo in rutas_modelos.items():
    print(f"\n Evaluando {nombre_modelo}...")
    
    # Cargar modelo
    modelo = load_model(ruta_modelo)
    
    # Predicción
    probabilidades = modelo.predict([X_mirna_val, X_trans_val])
    predicciones = (probabilidades > 0.5).astype(int).flatten()
    
    # Almacenar resultados
    resultados[nombre_modelo] = {
        'probabilidades': probabilidades,
        'predicciones': predicciones,
        'reporte': classification_report(y_val, predicciones, output_dict=True),
        'fpr': roc_curve(y_val, probabilidades)[0],
        'tpr': roc_curve(y_val, probabilidades)[1],
        'auc': auc(roc_curve(y_val, probabilidades)[0], roc_curve(y_val, probabilidades)[1])
    }
    
    # Mostrar métricas individuales
    print(f"\n Reporte de clasificación para {nombre_modelo}:")
    print(classification_report(y_val, predicciones, target_names=["NEGATIVO", "POSITIVO"]))

# Comparativa de métricas 
metricas_comparativas = ['precision', 'recall', 'f1-score']
for metrica in metricas_comparativas:
    print(f"\n🔹 {metrica.upper()}:")
    for modelo in resultados:
        valor = resultados[modelo]['reporte']['weighted avg'][metrica]
        print(f"{modelo}: {valor:.4f}")

# Gráfico comparativo de curvas ROC 
plt.figure(figsize=(8, 6))
for nombre, res in resultados.items():
    plt.plot(res['fpr'], res['tpr'], lw=2, 
             label=f"{nombre} (AUC = {res['auc']:.2f})")

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("Tasa de falsos positivos")
plt.ylabel("Tasa de verdaderos positivos")
plt.title("Comparación de Curvas ROC")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# Matrices de confusión comparativas 
fig, axes = plt.subplots(1, len(resultados), figsize=(12, 4))
fig.suptitle("Matrices de Confusión Comparativas")

for ax, (nombre, res) in zip(axes, resultados.items()):
    cm = confusion_matrix(y_val, res['predicciones'])
    ax.imshow(cm, cmap='Greens')
    ax.set_title(nombre)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["NEGATIVO", "POSITIVO"])
    ax.set_yticklabels(["NEGATIVO", "POSITIVO"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# === Métricas por clase (según tus resultados previos) ===
# Orden: [precision, recall, f1-score]
transformer_neg = [0.85, 0.92, 0.88]
transformer_pos = [0.92, 0.83, 0.87]

cnn_neg = [0.88, 0.88, 0.88]
cnn_pos = [0.88, 0.88, 0.88]

modelos = {
    "Modelo Transformers": {"NEGATIVO": transformer_neg, "POSITIVO": transformer_pos},
    "Modelo CNN": {"NEGATIVO": cnn_neg, "POSITIVO": cnn_pos}
}

metricas = ["Precisión", "Recall", "F1-score"]
x = np.arange(len(metricas))
width = 0.35

# === Graficar ===
fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, (nombre, datos) in zip(axs, modelos.items()):
    valores_neg = datos["NEGATIVO"]
    valores_pos = datos["POSITIVO"]

    barras1 = ax.bar(x - width/2, valores_neg, width, label="NEGATIVO", color="salmon")
    barras2 = ax.bar(x + width/2, valores_pos, width, label="POSITIVO", color="seagreen")

    ax.set_title(nombre)
    ax.set_xticks(x)
    ax.set_xticklabels(metricas)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("Valor")
    ax.legend()

    # Agregar valores encima de las barras
    for barras in [barras1, barras2]:
        for barra in barras:
            altura = barra.get_height()
            ax.text(barra.get_x() + barra.get_width()/2, altura + 0.01,
                    f"{altura:.2f}", ha="center", va="bottom", fontsize=9)

fig.suptitle("Métricas por clase para cada modelo (NEGATIVO vs. POSITIVO)", fontsize=14)
plt.tight_layout()
plt.show()

