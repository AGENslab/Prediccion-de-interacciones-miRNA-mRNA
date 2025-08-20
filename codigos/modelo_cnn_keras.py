import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate, LayerNormalization, MultiHeadAttention, Add
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import load_model
import tensorflow as tf

# datos
ruta_dataset = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/dataset_secuencias_balanceado.jsonl"
longitud_max_mirna = 30
longitud_max_transcrito = 300

# codificar
def codificar_secuencia(seq, maxlen):
    mapa = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
    codificada = np.zeros((maxlen, 4), dtype=np.float32)
    for i, base in enumerate(seq.upper()[:maxlen]):
        if base in mapa:
            codificada[i, mapa[base]] = 1.0
    return codificada

# cargar y procesar datos
X_mirna, X_transcrito, y = [], [], []
with open(ruta_dataset, "r", encoding="utf-8") as f:
    for linea in f:
        dato = json.loads(linea)
        X_mirna.append(codificar_secuencia(dato["secuencia_mirna"], longitud_max_mirna))
        X_transcrito.append(codificar_secuencia(dato["secuencia_transcrito"], longitud_max_transcrito))
        y.append(dato["etiqueta"])

X_mirna = np.array(X_mirna)
X_transcrito = np.array(X_transcrito)
y = np.array(y)

# separar entre entrenamiento y validacion
X_mirna_ent, X_mirna_val, X_trans_ent, X_trans_val, y_ent, y_val = train_test_split(
    X_mirna, X_transcrito, y, test_size=0.2, random_state=42, stratify=y
)

# crear modelo
def crear_modelo():
    entrada_mirna = Input(shape=(longitud_max_mirna, 4))
    entrada_trans = Input(shape=(longitud_max_transcrito, 4))

    # cnn para miRNA
    x1 = Conv1D(64, 3, activation='relu')(entrada_mirna)
    x1 = GlobalMaxPooling1D()(x1)

    # cnn mRNA
    x2 = Conv1D(64, 5, activation='relu')(entrada_trans)
    x2 = GlobalMaxPooling1D()(x2)

    # concatenar y aplicar atencion
    x = Concatenate()([x1, x2])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    salida = Dense(1, activation='sigmoid')(x)

    modelo = Model(inputs=[entrada_mirna, entrada_trans], outputs=salida)
    modelo.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy', AUC(name='AUC')])
    return modelo

modelo = crear_modelo()

#entrenar modelo

historial = modelo.fit(
    [X_mirna_ent, X_trans_ent], y_ent,
    validation_data=([X_mirna_val, X_trans_val], y_val),
    epochs=600,
    batch_size=32,
)

# guardar modelo
modelo.save("C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/modelo_cnn_attention_final.keras")

