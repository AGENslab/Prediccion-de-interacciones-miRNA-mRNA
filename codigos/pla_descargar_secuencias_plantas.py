import json
from Bio import Entrez, SeqIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# configuracion
Entrez.email = ""
Entrez.api_key = ""
ruta_dataset = "dataset_plantas_mirna.jsonl"
ruta_salida = "diccionario_gen_a_secuencia_plantas.json"

# cargar nombre de genes mrna

genes_unicos = set()
with open(ruta_dataset, "r", encoding="utf-8") as f:
    for linea in f:
        datos = json.loads(linea)
        genes_unicos.add(datos["gene_name"])
print(f"🔢 Total genes únicos a buscar: {len(genes_unicos)}")

#  obtener la secuencia de transcrito desde NCBI 
def obtener_secuencia_ncbi(gen_name):
    try:
        busqueda = Entrez.esearch(
            db="nucleotide",
            term=f"{gen_name}[Gene Name] AND plants[Organism] AND mRNA",
            retmax=1
        )
        resultado = Entrez.read(busqueda)
        if not resultado["IdList"]:
            return (gen_name, None)

        id_ncbi = resultado["IdList"][0]
        handle = Entrez.efetch(db="nucleotide", id=id_ncbi, rettype="fasta", retmode="text")
        seq_record = SeqIO.read(handle, "fasta")
        handle.close()

        secuencia = str(seq_record.seq).upper()
        if len(secuencia) > 400:
            secuencia = secuencia[:400]

        return (gen_name, secuencia)
    except Exception:
        return (gen_name, None)

# descarga paralela
diccionario_genes = {}
with ThreadPoolExecutor(max_workers=6) as executor:
    tareas = [executor.submit(obtener_secuencia_ncbi, gene) for gene in genes_unicos]
    for i, tarea in enumerate(as_completed(tareas), start=1):
        gene, secuencia = tarea.result()
        if secuencia:
            diccionario_genes[gene] = secuencia
        if i % 10 == 0:
            print(f"✅ {i}/{len(genes_unicos)} completados")

# guardar resultados
with open(ruta_salida, "w", encoding="utf-8") as f:
    json.dump(diccionario_genes, f, indent=2)

print(f"Secuencias guardadas en: {ruta_salida}")
print(f"Total genes con secuencia: {len(diccionario_genes)}")
