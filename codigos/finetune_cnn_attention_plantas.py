import json
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC

# configuracion
ruta_dataset_plantas = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/dataset_plantas_secuencias.jsonl"
ruta_modelo_humano = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/modelo_cnn_attention_final.keras"
ruta_modelo_ajustado = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/modelo_finetune_rechazo_plantas.keras"
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

# cargar datos de plantas
X_mirna, X_transcrito = [], []
with open(ruta_dataset_plantas, "r", encoding="utf-8") as f:
    for linea in f:
        d = json.loads(linea)
        X_mirna.append(codificar_secuencia(d["secuencia_mirna"], longitud_max_mirna))
        X_transcrito.append(codificar_secuencia(d["secuencia_transcrito"], longitud_max_transcrito))

X_mirna = np.array(X_mirna)
X_transcrito = np.array(X_transcrito)
y = np.ones(len(X_mirna))  

# carga modelo humano
modelo_base = load_model(ruta_modelo_humano)

# congelar capas base
for capa in modelo_base.layers:
    capa.trainable = False

# agregar cabezas a la salida
features = modelo_base.layers[-2].output

salida_clasificacion = Dense(1, activation="sigmoid", name="salida_clasificacion")(features)

recon_mirna = Dense(64, activation="relu", name="recon_mirna_dense")(features)
recon_mirna = Dense(longitud_max_mirna * 4, activation="sigmoid", name="recon_mirna_flat")(recon_mirna)
recon_mirna = Reshape((longitud_max_mirna, 4), name="salida_recon_mirna")(recon_mirna)

recon_trans = Dense(128, activation="relu", name="recon_trans_dense")(features)
recon_trans = Dense(longitud_max_transcrito * 4, activation="sigmoid", name="recon_trans_flat")(recon_trans)
recon_trans = Reshape((longitud_max_transcrito, 4), name="salida_recon_trans")(recon_trans)

modelo_finetune = Model(
    inputs=modelo_base.input,
    outputs=[salida_clasificacion, recon_mirna, recon_trans]
)

modelo_finetune.compile(
    optimizer=Adam(1e-5),
    loss=["binary_crossentropy", "mse", "mse"],
    loss_weights=[1.0, 0.5, 0.5],
    metrics={"salida_clasificacion": ["accuracy", AUC(name="AUC")]}
)

# entrenamiento
modelo_finetune.fit(
    [X_mirna, X_transcrito], [y, X_mirna, X_transcrito],
    epochs=300,
    batch_size=32,
    callbacks=[EarlyStopping(monitor="salida_clasificacion_auc", patience=5, restore_best_weights=True, mode="max")]
)

modelo_finetune.save(ruta_modelo_ajustado)
print(f" Modelo ajustado guardado en: {ruta_modelo_ajustado}")
