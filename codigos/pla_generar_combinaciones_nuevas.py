import json
import random
from collections import defaultdict

# rutas
ruta_original = "dataset_plantas_secuencias.jsonl"
ruta_salida = "dataset_combinaciones_artificiales.jsonl"

# datos
interacciones_existentes = set()
miRNAs_por_especie = defaultdict(dict)  
genes_por_especie = defaultdict(dict)   

with open(ruta_original, "r", encoding="utf-8") as f:
    for linea in f:
        d = json.loads(linea)
        especie = d["especie"]
        mirna = d["mirna"]
        gene = d["gene_name"]
        
        miRNAs_por_especie[especie][mirna] = d["secuencia_mirna"]
        genes_por_especie[especie][gene] = d["secuencia_transcrito"]
        interacciones_existentes.add((mirna, gene, especie))

# generar todas las combinaciones posibles por especie 
nuevos_pares = []
for especie in miRNAs_por_especie:
    print(f"🌿 Procesando especie: {especie}")
    for mirna, sec_mirna in miRNAs_por_especie[especie].items():
        for gene, sec_trans in genes_por_especie[especie].items():
            if (mirna, gene, especie) not in interacciones_existentes:
                nuevos_pares.append({
                    "mirna": mirna,
                    "gene_name": gene,
                    "especie": especie,
                    "secuencia_mirna": sec_mirna,
                    "secuencia_transcrito": sec_trans
                })

print(f" Total de combinaciones nuevas generadas: {len(nuevos_pares)}")

# guardar archivo
with open(ruta_salida, "w", encoding="utf-8") as f:
    for par in nuevos_pares:
        json.dump(par, f)
        f.write("\n")

print(" Archivo generado:", ruta_salida)
print(f"Total de combinaciones guardadas: {len(nuevos_pares)}")
