import pandas as pd
import json
from Bio import SeqIO
from tqdm import tqdm

# === Rutas ===
ruta_tarbase = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/tarbase_data.csv"
ruta_mirtarbase = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/miRTarBase_MTI.xlsx"
ruta_mature = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/mature.fa"
ruta_diccionario = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/diccionario_gen_a_secuencia.json"
salida_jsonl = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/dataset_secuencias_final.jsonl"

# === Cargar secuencias de miRNA ===
print("🧬 Cargando miRNAs...")
mirnas = {}
for registro in SeqIO.parse(ruta_mature, "fasta"):
    if registro.id.startswith("hsa"):
        mirnas[registro.id] = str(registro.seq)

# === Cargar diccionario gene → secuencia ===
print("📚 Cargando secuencias de genes desde NCBI...")
with open(ruta_diccionario, "r", encoding="utf-8") as f:
    diccionario_genes = json.load(f)

# === Cargar datasets ===
print("📄 Leyendo TarBase y miRTarBase...")
df_tarbase = pd.read_csv(ruta_tarbase, sep="\t")
df_mirtarbase = pd.read_excel(ruta_mirtarbase)

# === Filtrar humanos ===
df_tarbase = df_tarbase[df_tarbase["species"] == "Homo sapiens"]
df_mirtarbase = df_mirtarbase[df_mirtarbase["Species (Target Gene)"] == "Homo sapiens"]

# === Extraer interacciones ===
interacciones = set()

print("🔍 Procesando TarBase...")
for _, fila in df_tarbase.iterrows():
    mirna = fila["mirna"]
    gen = fila["geneName"]
    if pd.notna(mirna) and pd.notna(gen):
        interacciones.add((mirna.strip(), gen.strip(), 1 if fila["positive_negative"] == "POSITIVE" else 0))

print("🔍 Procesando miRTarBase...")
for _, fila in df_mirtarbase.iterrows():
    mirna = fila["miRNA"]
    gen = fila["Target Gene"]
    if pd.notna(mirna) and pd.notna(gen):
        interacciones.add((mirna.strip(), gen.strip(), 1))

print(f"🔢 Total de interacciones crudas: {len(interacciones)}")

# === Generar dataset con secuencias ===
print("⚙️ Generando dataset con secuencias válidas...")
conteo_validas = 0
with open(salida_jsonl, "w", encoding="utf-8") as salida:
    for mirna, gen, etiqueta in tqdm(interacciones):
        secuencia_mirna = mirnas.get(mirna)
        secuencia_gen = diccionario_genes.get(gen)
        if secuencia_mirna and secuencia_gen:
            salida.write(json.dumps({
                "mirna": mirna,
                "gene_name": gen,
                "secuencia_mirna": secuencia_mirna,
                "secuencia_transcrito": secuencia_gen,
                "etiqueta": etiqueta
            }) + "\n")
            conteo_validas += 1

print(f"✅ Dataset generado con {conteo_validas} pares válidos.")
