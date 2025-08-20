import pandas as pd
import json

# rutas
ruta_tarbase = "tarbase_data.csv"
ruta_mirtarbase = "miRTarBase_MTI.xlsx"
ruta_salida = "dataset_plantas.jsonl"

# especies presentes
especies_vegetales = {
    "Arabidopsis thaliana",
    "Glycine max",
    "Medicago truncatula",
    "Oryza sativa",
    "Physcomitrella patens",
    "Solanum lycopersicum",
    "Vitis vinifera",
    "Zea mays"
}

# cargar tarbase
df_tarbase = pd.read_csv(ruta_tarbase, sep="\t")
df_tarbase = df_tarbase[df_tarbase["species"].isin(especies_vegetales)]
df_tarbase = df_tarbase[["mirna", "geneName", "positive_negative", "species"]]
df_tarbase = df_tarbase.rename(columns={"geneName": "gen", "positive_negative": "etiqueta"})
df_tarbase["fuente"] = "TarBase"
df_tarbase["etiqueta"] = df_tarbase["etiqueta"].astype(str).str.strip().str.upper()
df_tarbase = df_tarbase[df_tarbase["etiqueta"].isin(["POSITIVE", "NEGATIVE"])]

# cargar mirbase

df_mirtar = pd.read_excel(ruta_mirtarbase)
df_mirtar = df_mirtar[df_mirtar["Species (Target Gene)"].isin(especies_vegetales)]
df_mirtar = df_mirtar[["miRNA", "Target Gene", "Support Type", "Species (Target Gene)"]]
df_mirtar = df_mirtar.rename(columns={
    "miRNA": "mirna",
    "Target Gene": "gen",
    "Support Type": "etiqueta",
    "Species (Target Gene)": "species"
})
df_mirtar["fuente"] = "miRTarBase"
df_mirtar["etiqueta"] = df_mirtar["etiqueta"].map({"Functional MTI": 1, "Non-Functional MTI": 0})

# unir dataset y exportar

df_final = pd.concat([df_tarbase, df_mirtar], ignore_index=True)
df_final = df_final.dropna(subset=["mirna", "gen", "etiqueta"])

with open(ruta_salida, "w", encoding="utf-8") as f:
    for _, fila in df_final.iterrows():
        etiqueta = (
            1 if str(fila["etiqueta"]).strip().upper() == "POSITIVE"
            else 0 if str(fila["etiqueta"]).strip().upper() == "NEGATIVE"
            else int(fila["etiqueta"])  
        )
        json.dump({
            "mirna": fila["mirna"],
            "gene_name": fila["gen"],
            "etiqueta": etiqueta,
            "especie": fila["species"],
            "fuente": fila["fuente"]
        }, f)
        f.write("\n")

print(f" Dataset generado: {ruta_salida}")
print(f"Total de pares vegetales: {len(df_final)}")
