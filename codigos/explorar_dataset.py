from Bio import SeqIO
import pandas as pd

# === 1. Revisar mature.fa
print("\n🧬 Revisando mature.fa:")
ruta_mature = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/mature.fa"
for i, record in enumerate(SeqIO.parse(ruta_mature, "fasta")):
    if record.id.startswith("hsa"):
        print(f"- {record.id}: {record.seq}")
    if i == 9:
        break

# === 2. Revisar cdna.all.fa
print("\n📚 Revisando Homo_sapiens.GRCh38.cdna.all.fa:")
ruta_cdna = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/Homo_sapiens.GRCh38.cdna.all.fa"
for i, record in enumerate(SeqIO.parse(ruta_cdna, "fasta")):
    print(f"- {record.id.split('|')[0]}")
    if i == 9:
        break

# === 3. Revisar GTF
print("\n📄 Mapeos gene_name → transcript_id desde GTF:")
ruta_gtf = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/Homo_sapiens.GRCh38.113.gtf"
conteo = 0
with open(ruta_gtf, encoding="utf-8") as archivo:
    for linea in archivo:
        if "\ttranscript\t" not in linea or linea.startswith("#"):
            continue
        campos = linea.strip().split("\t")
        info = campos[8]
        atributos = {}
        for campo in info.strip().split(";"):
            partes = campo.strip().split()
            if len(partes) >= 2:
                atributos[partes[0]] = partes[1].strip('"')
        if "gene_name" in atributos and "transcript_id" in atributos:
            print(f'- {atributos["gene_name"]} → {atributos["transcript_id"]}')
            conteo += 1
        if conteo >= 10:
            break

# === 4. Revisar miRTarBase_MTI.xlsx
print("\n📑 Revisando miRTarBase_MTI.xlsx:")
ruta_excel = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/miRTarBase_MTI.xlsx"
df_excel = pd.read_excel(ruta_excel)
print(df_excel.head(10).to_string(index=False))

# === 5. Revisar tarbase_data.csv
print("\n📄 Revisando tarbase_data.csv:")
ruta_tarbase = "C:/Users/Jaime Escobar/Desktop/Tesispy/env_nuevo/data/tarbase_data.csv"
df_tarbase = pd.read_csv(ruta_tarbase, sep="\\t", encoding="utf-8")

print(f"✅ Total de filas: {len(df_tarbase)}")
print("🔎 Columnas disponibles:")
for col in df_tarbase.columns:
    print(f"- {col}")

print("\n🧪 Primeras 10 filas de TarBase:")
print(df_tarbase.head(10).to_string(index=False))
