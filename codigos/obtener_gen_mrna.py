import pandas as pd
from Bio import Entrez
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# configuracion
Entrez.email = ""
Entrez.api_key = ""

ruta_tarbase = "tarbase_data.csv"
ruta_mirtarbase = "miRTarBase_MTI.xlsx"

#leer datos

df_tarbase = pd.read_csv(ruta_tarbase, sep="\t")
df_mirtarbase = pd.read_excel(ruta_mirtarbase)

# filtrar humanos y extraer genes

genes_tarbase = df_tarbase[df_tarbase["species"] == "Homo sapiens"]["geneName"].dropna().unique()
genes_mirtarbase = df_mirtarbase[df_mirtarbase["Species (Target Gene)"] == "Homo sapiens"]["Target Gene"].dropna().unique()

# unir y eliminar duplicados
todos_los_genes = sorted(set(genes_tarbase) | set(genes_mirtarbase))
print(f"🔢 Total de genes humanos únicos encontrados: {len(todos_los_genes)}")

#  obtener secuencia desde NCBI 
def obtener_secuencia_ncbi(nombre_gen):
    try:
        busqueda = Entrez.esearch(db="gene", term=f"{nombre_gen}[Gene Name] AND Homo sapiens[Organism]", retmax=1)
        resultado = Entrez.read(busqueda)
        if not resultado["IdList"]:
            return (nombre_gen, None)

        gene_id = resultado["IdList"][0]
        detalles = Entrez.efetch(db="gene", id=gene_id, retmode="xml")
        gene_data = Entrez.read(detalles)

        for feature in gene_data[0].get("Entrezgene_locus", []):
            if "Gene-commentary_products" in feature:
                for producto in feature["Gene-commentary_products"]:
                    if "Gene-commentary_accession" in producto:
                        acc = producto["Gene-commentary_accession"]
                        seq_handle = Entrez.efetch(db="nucleotide", id=acc, rettype="fasta", retmode="text")
                        fasta = seq_handle.read()
                        return (nombre_gen, fasta)
    except Exception as e:
        return (nombre_gen, None)

    return (nombre_gen, None)

# ejecutar en paralelo
diccionario_genes = {}
with ThreadPoolExecutor(max_workers=6) as executor:
    futuros = [executor.submit(obtener_secuencia_ncbi, gen) for gen in todos_los_genes]
    for i, future in enumerate(as_completed(futuros)):
        gen, seq = future.result()
        if seq:
            diccionario_genes[gen] = seq
        if (i + 1) % 10 == 0:
            print(f"✅ {i + 1}/{len(todos_los_genes)} completados")

# guardar resultados

with open("C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/diccionario_gen_a_secuencia.json", "w", encoding="utf-8") as f:
    json.dump(diccionario_genes, f, indent=2)

with open("C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/diccionario_gen_a_secuencia.csv", "w", encoding="utf-8") as f:
    f.write("gene_name,secuencia_fasta\n")
    for gen, seq in diccionario_genes.items():
        seq_limpia = seq.replace("\n", " ").replace(",", "")
        f.write(f"{gen},\"{seq_limpia}\"\n")

print(" Diccionario guardado con", len(diccionario_genes), "genes.")
