import pandas as pd

# rutas
ruta_tarbase = "tarbase_data.csv"
ruta_mirtarbase = "miRTarBase_MTI.xlsx"

# leer tarbase
df_tarbase = pd.read_csv(ruta_tarbase, sep="\t", low_memory=False)
especies_tarbase = df_tarbase["species"].dropna().unique()
print(f" Especies únicas en TarBase: {len(especies_tarbase)}")
for especie in sorted(especies_tarbase):
    print(" -", especie)

# leer mirbase

df_mirtarbase = pd.read_excel(ruta_mirtarbase)
if "Species (Target Gene)" in df_mirtarbase.columns:
    especies_mirtarbase = df_mirtarbase["Species (Target Gene)"].dropna().unique()
    print(f" Especies únicas en miRTarBase: {len(especies_mirtarbase)}")
    for especie in sorted(especies_mirtarbase):
        print(" -", especie)
else:
    print(" No se encontró la columna 'Species (Target Gene)' en miRTarBase.")
