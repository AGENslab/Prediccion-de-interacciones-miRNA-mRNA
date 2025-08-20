import json

# rutas
ruta_entrada = "dataset_plantas_mirna.jsonl"
ruta_diccionario = "diccionario_gen_a_secuencia_plantas.json"
ruta_salida = "dataset_plantas_secuencias.jsonl"

# cargar secuencias 
with open(ruta_diccionario, "r", encoding="utf-8") as f:
    diccionario_genes = json.load(f)

# procesar dataset original
pares_validos = 0
with open(ruta_entrada, "r", encoding="utf-8") as entrada, open(ruta_salida, "w", encoding="utf-8") as salida:
    for linea in entrada:
        datos = json.loads(linea)
        gene = datos["gene_name"]

        sec_transcrito = diccionario_genes.get(gene)
        if sec_transcrito:
            datos["secuencia_transcrito"] = sec_transcrito
            json.dump(datos, salida)
            salida.write("\n")
            pares_validos += 1

print(f" Archivo generado: {ruta_salida}")
print(f"Total de pares con secuencia de transcrito: {pares_validos}")
