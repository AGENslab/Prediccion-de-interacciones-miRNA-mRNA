import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Configuración 
rutas_modelos = {
    "Modelo Transformers + Self Attention": "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/modelo_finetune_transformer_plantas.keras",
    "Modelo CNN + Self Attention": "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/modelo_finetune_rechazo_plantas.keras"
}
ruta_dataset = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/plantas_pos_neg_balanceado.jsonl"
longitud_max_mirna = 30
longitud_max_transcrito = 300
umbral_rechazo = 0.15

# Función para codificar secuencia 
def codificar_secuencia(seq, maxlen):
    mapa = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
    codificada = np.zeros((maxlen, 4), dtype=np.float32)
    for i, base in enumerate(seq.upper()[:maxlen]):
        if base in mapa:
            codificada[i, mapa[base]] = 1.0
    return codificada

# Cargar datos
X_mirna, X_trans, y_true = [], [], []
with open(ruta_dataset, "r", encoding="utf-8") as f:
    for linea in f:
        d = json.loads(linea)
        X_mirna.append(codificar_secuencia(d["secuencia_mirna"], longitud_max_mirna))
        X_trans.append(codificar_secuencia(d["secuencia_transcrito"], longitud_max_transcrito))
        y_true.append(d.get("etiqueta", 1))  # Asumimos que todos son positivos (1)

X_mirna = np.array(X_mirna)
X_trans = np.array(X_trans)
y_true = np.array(y_true)

# Evaluación de modelos 
resultados = {}

for nombre_modelo, ruta_modelo in rutas_modelos.items():
    
    # Cargar modelo
    modelo = load_model(ruta_modelo)
    
    # Predicciones y reconstrucciones
    preds, recon_mirna, recon_trans = modelo.predict([X_mirna, X_trans], batch_size=32)
    
    # Cálculo de errores
    error_mirna = np.mean((X_mirna - recon_mirna)**2, axis=(1, 2))
    error_trans = np.mean((X_trans - recon_trans)**2, axis=(1, 2))
    error_total = (error_mirna + error_trans) / 2
    
    # Aplicar umbral de rechazo
    mascara_aceptados = error_total < umbral_rechazo
    preds_filtradas = np.where(mascara_aceptados, preds.flatten(), np.nan)
    
    # Estadísticas de rechazo
    total_ejemplos = len(preds)
    aceptados = np.sum(mascara_aceptados)
    rechazados = total_ejemplos - aceptados
    
    # Métricas para ejemplos aceptados
    y_pred = (preds_filtradas[mascara_aceptados] > 0.5).astype(int)
    y_true_aceptados = y_true[mascara_aceptados]
    
    # Calcular métricas
    precision = precision_score(y_true_aceptados, y_pred, zero_division=0)
    recall = recall_score(y_true_aceptados, y_pred, zero_division=0)
    f1 = f1_score(y_true_aceptados, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true_aceptados, y_pred)
    falsos_negativos = np.sum((y_pred == 0) & (y_true_aceptados == 1))
    
    # Almacenar resultados
    resultados[nombre_modelo] = {
        'total_ejemplos': total_ejemplos,
        'aceptados': aceptados,
        'rechazados': rechazados,
        'tasa_rechazo': rechazados / total_ejemplos,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'falsos_negativos': falsos_negativos,
        'tasa_falsos_negativos': falsos_negativos / np.sum(y_true_aceptados == 1),
        'predicciones': preds_filtradas,
        'error_total': error_total
    }
    
    # Mostrar resultados
    print(f" Estadísticas para {nombre_modelo}:")
    print(f"  • Total de ejemplos: {total_ejemplos}")
    print(f"  • Ejemplos aceptados: {aceptados} ({aceptados/total_ejemplos:.1%})")
    print(f"  • Ejemplos rechazados: {rechazados} ({rechazados/total_ejemplos:.1%})")
    print(f"\n Métricas (solo aceptados):")
    print(f"  • Precisión: {precision:.4f}")
    print(f"  • Recall/Sensibilidad: {recall:.4f}")
    print(f"  • F1-score: {f1:.4f}")
    print(f"  • Exactitud: {accuracy:.4f}")
    print(f"  • Falsos negativos: {falsos_negativos}")
    print(f"  • Tasa falsos negativos: {resultados[nombre_modelo]['tasa_falsos_negativos']:.4f}")

# Visualización comparativa 

# 1. Comparación de tasas de rechazo y métricas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico de tasas de rechazo
tasas_rechazo = [res['tasa_rechazo'] for res in resultados.values()]
ax1.bar(resultados.keys(), tasas_rechazo, color=['skyblue', 'salmon'])
ax1.set_title('Tasas de Rechazo')
ax1.set_ylabel('Tasa de rechazo')
ax1.set_ylim(0, 1)

# Gráfico de métricas
metricas = ['precision', 'recall', 'f1', 'accuracy']
x = np.arange(len(metricas))
width = 0.35

for i, (nombre, res) in enumerate(resultados.items()):
    valores = [res[metrica] for metrica in metricas]
    ax2.bar(x + i*width, valores, width, label=nombre)

ax2.set_title('Comparación de Métricas (aceptados)')
ax2.set_xticks(x + width/2)
ax2.set_xticklabels(metricas)
ax2.legend()
ax2.set_ylim(0, 1.1)

plt.tight_layout()
plt.show()

# 2. Comparación de falsos negativos
plt.figure(figsize=(10, 6))
nombres = list(resultados.keys())
falsos_neg = [res['falsos_negativos'] for res in resultados.values()]
tasa_fn = [res['tasa_falsos_negativos'] for res in resultados.values()]

x = np.arange(len(nombres))
width = 0.35

plt.bar(x - width/2, falsos_neg, width, label='Falsos negativos', color='salmon')
plt.bar(x + width/2, tasa_fn, width, label='Tasa falsos negativos', color='lightcoral')

plt.xlabel('Modelos')
plt.ylabel('Cantidad/Tasa')
plt.title('Comparación de Falsos Negativos')
plt.xticks(x, nombres)
plt.legend()
plt.tight_layout()
plt.show()

# 3. Distribución de errores de reconstrucción
plt.figure(figsize=(10, 6))
for nombre, res in resultados.items():
    sns.kdeplot(res['error_total'], label=nombre, alpha=0.7)
plt.axvline(umbral_rechazo, color='red', linestyle='--', label='Umbral de rechazo')
plt.title("Distribución de Errores de Reconstrucción")
plt.xlabel("Error de reconstrucción")
plt.ylabel("Densidad")
plt.legend()
plt.tight_layout()
plt.show()

# 4. Distribución de predicciones aceptadas
plt.figure(figsize=(10, 6))
for nombre, res in resultados.items():
    preds_aceptadas = res['predicciones'][~np.isnan(res['predicciones'])]
    sns.kdeplot(preds_aceptadas, label=nombre, alpha=0.7)
plt.axvline(0.5, color='red', linestyle='--', label='Umbral de decisión')
plt.title("Distribución de Scores de Predicción (aceptados)")
plt.xlabel("Score de predicción")
plt.ylabel("Densidad")
plt.legend()
plt.tight_layout()
plt.show()