import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

# === Configuración ===
ruta_modelo = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/modelo_finetune_rechazo_plantas.keras"
ruta_dataset = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/dataset_plantas_secuencias.jsonl"
longitud_max_mirna = 30
longitud_max_transcrito = 300
umbral_rechazo = 0.15

# === Función para codificar secuencia ===
def codificar_secuencia(seq, maxlen):
    mapa = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
    codificada = np.zeros((maxlen, 4), dtype=np.float32)
    for i, base in enumerate(seq.upper()[:maxlen]):
        if base in mapa:
            codificada[i, mapa[base]] = 1.0
    return codificada

# === Cargar datos ===
print("📄 Cargando dataset enriquecido de plantas...")
X_mirna, X_trans = [], []
with open(ruta_dataset, "r", encoding="utf-8") as f:
    for linea in f:
        d = json.loads(linea)
        X_mirna.append(codificar_secuencia(d["secuencia_mirna"], longitud_max_mirna))
        X_trans.append(codificar_secuencia(d["secuencia_transcrito"], longitud_max_transcrito))

X_mirna = np.array(X_mirna)
X_trans = np.array(X_trans)

# === Cargar modelo ===
print("📦 Cargando modelo ajustado...")
modelo = load_model(ruta_modelo)

# === Predicciones con mecanismo de rechazo ===
print("🧠 Realizando predicciones con mecanismo de rechazo...")
preds, recon_mirna, recon_trans = modelo.predict([X_mirna, X_trans], batch_size=32)

# === Cálculo de error de reconstrucción promedio ===
error_mirna = np.mean((X_mirna - recon_mirna)**2, axis=(1, 2))
error_trans = np.mean((X_trans - recon_trans)**2, axis=(1, 2))
error_total = (error_mirna + error_trans) / 2

# === Aplicar umbral de rechazo ===
preds_filtradas = np.where(error_total < umbral_rechazo, preds.flatten(), 0)

# === Resultados ===
print("\n📊 Resultados:")
print(f"🔢 Total de ejemplos: {len(preds)}")
print(f"✅ Aceptados por el modelo: {np.sum(error_total < umbral_rechazo)}")
print(f"❌ Rechazados por error alto: {np.sum(error_total >= umbral_rechazo)}")
print(f"🌱 Predicciones positivas (score > 0.5): {np.sum(preds_filtradas > 0.5)}")

# === Visualización ===
plt.figure(figsize=(8, 6))
sns.histplot(error_total, bins=30, kde=True)
plt.axvline(umbral_rechazo, color='red', linestyle='--', label='Umbral de rechazo')
plt.title("Distribución del Error de Reconstrucción")
plt.xlabel("Error de reconstrucción")
plt.ylabel("Frecuencia")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(preds_filtradas[preds_filtradas > 0], bins=20, kde=True)
plt.title("Distribución de Scores de Predicción (aceptados)")
plt.xlabel("Score de predicción")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()
