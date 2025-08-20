import json
import random

ruta_entrada = "dataset_secuencias_final.jsonl"
ruta_salida = "dataset_secuencias_balanceado.jsonl"

#cargar dataset
positivos, negativos = [], []

with open(ruta_entrada, "r", encoding="utf-8") as f:
    for linea in f:
        dato = json.loads(linea)
        if dato["etiqueta"] == 1:
            positivos.append(dato)
        else:
            negativos.append(dato)

print(f" Positivos: {len(positivos)}")
print(f" Negativos: {len(negativos)}")

# elegir minimo para balancear
limite = min(len(positivos), len(negativos))
random.shuffle(positivos)
random.shuffle(negativos)

positivos = positivos[:limite]
negativos = negativos[:limite]

datos_balanceados = positivos + negativos
random.shuffle(datos_balanceados)

print(f" Total balanceado: {len(datos_balanceados)} pares")

with open(ruta_salida, "w", encoding="utf-8") as f:
    for d in datos_balanceados:
        f.write(json.dumps(d) + "\n")

print(f" Dataset balanceado guardado en: {ruta_salida}")
