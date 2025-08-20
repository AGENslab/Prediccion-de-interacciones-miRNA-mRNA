import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
import tensorflow as tf

# === Configuración ===
RUTA_DATOS = "dataset_secuencias_balanceado.jsonl"
RUTA_SALIDA_MODELO = "modelo_transformer_f.keras"
LONG_MIRNA = 30
LONG_TRANSCRITO = 300

# === Codificador One-Hot ===
def codificar_secuencia(seq, maxlen):
    mapa = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
    codificada = np.zeros((maxlen, 4), dtype=np.float32)
    for i, base in enumerate(seq.upper()[:maxlen]):
        if base in mapa:
            codificada[i, mapa[base]] = 1.0
    return codificada

# === Cargar Datos ===
print("📦 Cargando datos...")
X_mirna, X_trans, y = [], [], []
with open(RUTA_DATOS, "r", encoding="utf-8") as f:
    for linea in f:
        d = json.loads(linea)
        X_mirna.append(codificar_secuencia(d["secuencia_mirna"], LONG_MIRNA))
        X_trans.append(codificar_secuencia(d["secuencia_transcrito"], LONG_TRANSCRITO))
        y.append(d["etiqueta"])

X_mirna = np.array(X_mirna)
X_trans = np.array(X_trans)
y = np.array(y)

# === Separar Conjunto de Validación ===
X_mirna_ent, X_mirna_val, X_trans_ent, X_trans_val, y_ent, y_val = train_test_split(
    X_mirna, X_trans, y, test_size=0.2, stratify=y, random_state=42
)

# === Crear Modelo ===
def crear_modelo_transformer():
    entrada_mirna = Input(shape=(LONG_MIRNA, 4), name="entrada_mirna")
    entrada_trans = Input(shape=(LONG_TRANSCRITO, 4), name="entrada_transcrito")

    # Proyección inicial
    x1 = Conv1D(64, 3, activation="relu", padding="same")(entrada_mirna)
    x2 = Conv1D(128, 5, activation="relu", padding="same")(entrada_trans)

    # Self-Attention por separado
    att1 = MultiHeadAttention(num_heads=4, key_dim=32)(x1, x1)
    att1 = Add()([x1, att1])
    att1 = LayerNormalization()(att1)
    att1 = GlobalMaxPooling1D()(att1)

    att2 = MultiHeadAttention(num_heads=4, key_dim=32)(x2, x2)
    att2 = Add()([x2, att2])
    att2 = LayerNormalization()(att2)
    att2 = GlobalMaxPooling1D()(att2)

    # Fusionar
    fusionado = Concatenate()([att1, att2])
    x = Dense(128, activation="relu")(fusionado)
    x = Dropout(0.3)(x)
    salida = Dense(1, activation="sigmoid", name="salida")(x)

    modelo = Model(inputs=[entrada_mirna, entrada_trans], outputs=salida)
    modelo.compile(
        optimizer=Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", AUC(name="AUC")]
    )
    return modelo

# === Entrenar ===
print("🚀 Entrenando modelo...")
modelo = crear_modelo_transformer()
modelo.fit(
    [X_mirna_ent, X_trans_ent], y_ent,
    validation_data=([X_mirna_val, X_trans_val], y_val),
    epochs=300,
    batch_size=32,
    callbacks=[EarlyStopping(monitor="val_auc", patience=5, mode="max", restore_best_weights=True)]
)

# === Guardar Modelo ===
modelo.save(RUTA_SALIDA_MODELO)
print(f"✅ Modelo guardado en: {RUTA_SALIDA_MODELO}")

