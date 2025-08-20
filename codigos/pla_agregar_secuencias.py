import json
from Bio import SeqIO

# rutas
ruta_dataset = "dataset_plantas.jsonl"
ruta_mature = "mature.fa"
ruta_salida = "dataset_plantas_mirna.jsonl"

# cargar secuencias de miRNA desde mature.fa 

dicc_mirnas = {
    record.id.split()[0]: str(record.seq).upper()
    for record in SeqIO.parse(ruta_mature, "fasta")
}

# procesar dataset y agregar secuencias de miRNA 

total = 0
encontrados = 0

with open(ruta_dataset, "r", encoding="utf-8") as entrada, open(ruta_salida, "w", encoding="utf-8") as salida:
    for linea in entrada:
        total += 1
        datos = json.loads(linea)
        mirna = datos["mirna"]
        secuencia = dicc_mirnas.get(mirna)
        
        if secuencia:
            datos["secuencia_mirna"] = secuencia
            json.dump(datos, salida)
            salida.write("\n")
            encontrados += 1

print(f"Archivo generado: {ruta_salida}")
print(f"Pares totales procesados: {total}")
print(f"Pares con miRNA encontrado: {encontrados}")
print(f"Pares descartados por falta de miRNA: {total - encontrados}")
