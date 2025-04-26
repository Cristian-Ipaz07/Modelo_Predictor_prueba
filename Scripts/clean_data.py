import pandas as pd
import os
from datetime import datetime

# Rutas
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Cargar dataset principal
csv_path = os.path.join(DATA_DIR, "2024-20252.csv")
df = pd.read_csv(csv_path)

# ---- 1. FILTRAR SOLO PARTIDOS TEMPORADA 2024-2025 ----
# Asegúrate que haya una columna de fecha reconocible (ej. 'Date')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df_before_filtering = df.copy()  # Copia antes del filtrado para reporte
df = df[df['Date'].dt.year >= 2024]  # Temporada actual

# ---- 2. LIMPIEZA GENERAL ----
# Eliminar duplicados
duplicates_before = df_before_filtering.duplicated().sum()
df = df.drop_duplicates()

# Opcional: columnas irrelevantes (puedes ajustar esta lista)
irrelevant_cols = ['Unnamed: 0']  # Ajusta según las columnas que veas
removed_columns = [col for col in irrelevant_cols if col in df.columns]

df_before_cleaning = df.copy()  # Copia antes de limpieza para reporte
df = df.drop(columns=[col for col in removed_columns if col in df.columns], errors='ignore')

# Rellenar valores nulos si es razonable
null_count_before = df_before_cleaning.isnull().sum().sum()
df = df.fillna(0)

# ---- 3. GUARDAR ARCHIVO LIMPIO ----
today = datetime.now().strftime("%Y-%m-%d")
output_path = os.path.join(RAW_DIR, f"{today}_clean.csv")
df.to_csv(output_path, index=False)

# Resumen de cambios
print(f"✅ Dataset limpio guardado en: {output_path}")
print("\nResumen de cambios realizados:")

# Resumen de filtrado por fecha
print(f"- Filtrado de temporada 2024-2025: {len(df_before_filtering) - len(df)} filas eliminadas.")

# Resumen de eliminación de duplicados
print(f"- Eliminación de duplicados: {duplicates_before} duplicados eliminados.")

# Resumen de eliminación de columnas
if removed_columns:
    print(f"- Columnas eliminadas: {', '.join(removed_columns)}")
else:
    print("- No se eliminaron columnas irrelevantes.")

# Resumen de valores nulos
print(f"- Valores nulos antes de limpiar: {null_count_before} valores nulos.")
print(f"- Valores nulos después de limpiar: {df.isnull().sum().sum()} valores nulos restantes.")
