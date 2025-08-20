import json
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
import tensorflow as tf

# configuracion
ruta_dataset_plantas = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/dataset_plantas_secuencias.jsonl"
ruta_modelo_humano = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/modelo_transformer_f.keras"
ruta_modelo_ajustado = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/modelo_finetune_transformer_plantas.keras"
longitud_max_mirna = 30
longitud_max_transcrito = 300

# codificacion
def codificar_secuencia(seq, maxlen):
    mapa = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
    codificada = np.zeros((maxlen, 4), dtype=np.float32)
    for i, base in enumerate(seq.upper()[:maxlen]):
        if base in mapa:
            codificada[i, mapa[base]] = 1.0
    return codificada

# dataset plantas
X_mirna, X_transcrito = [], []
with open(ruta_dataset_plantas, "r", encoding="utf-8") as f:
    for linea in f:
        d = json.loads(linea)
        X_mirna.append(codificar_secuencia(d["secuencia_mirna"], longitud_max_mirna))
        X_transcrito.append(codificar_secuencia(d["secuencia_transcrito"], longitud_max_transcrito))

X_mirna = np.array(X_mirna)
X_transcrito = np.array(X_transcrito)
y = np.ones(len(X_mirna))  

#modelo humano

modelo_base = load_model(ruta_modelo_humano)

#congelar las capas de modelo base

for capa in modelo_base.layers:
    capa.trainable = False

# añadir nuevas salidas
features = modelo_base.layers[-2].output  

# clasificacion
salida_clasificacion = Dense(1, activation="sigmoid", name="salida_clasificacion")(features)

# recontruccion miRNA
recon_mirna = Dense(64, activation="relu", name="recon_mirna_dense")(features)
recon_mirna = Dense(longitud_max_mirna * 4, activation="sigmoid", name="recon_mirna_flat")(recon_mirna)
recon_mirna = Reshape((longitud_max_mirna, 4), name="salida_recon_mirna")(recon_mirna)

# reconstruccion del transcrito
recon_trans = Dense(128, activation="relu", name="recon_trans_dense")(features)
recon_trans = Dense(longitud_max_transcrito * 4, activation="sigmoid", name="recon_trans_flat")(recon_trans)
recon_trans = Reshape((longitud_max_transcrito, 4), name="salida_recon_trans")(recon_trans)

# modelo completo
modelo_finetune = Model(
    inputs=modelo_base.input,
    outputs=[salida_clasificacion, recon_mirna, recon_trans]
)

# compilar
modelo_finetune.compile(
    optimizer=Adam(1e-5),
    loss=["binary_crossentropy", "mse", "mse"],
    loss_weights=[1.0, 0.5, 0.5],
    metrics={"salida_clasificacion": ["accuracy", AUC(name="AUC")]}
)

# entrenar modelo
modelo_finetune.fit(
    [X_mirna, X_transcrito], [y, X_mirna, X_transcrito],
    epochs=300,
    batch_size=32,
    callbacks=[EarlyStopping(monitor="salida_clasificacion_auc", patience=5, restore_best_weights=True, mode="max")]
)

# guardar modelo
modelo_finetune.save(ruta_modelo_ajustado)
print(f"Modelo ajustado guardado en: {ruta_modelo_ajustado}")
