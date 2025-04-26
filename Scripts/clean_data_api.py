import pandas as pd
import os
from datetime import datetime
from balldontlie import BalldontlieAPI

# Directorio de almacenamiento
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Inicializar la API con la clave de API
api_key = '24bd3cf8-ae40-44c3-beb1-6fbf324f41f0'
api = BalldontlieAPI(api_key=api_key)

# Función para obtener los datos de la API de la NBA para toda la temporada
def get_api_data():
    all_games = []
    
    try:
        # Llamada a la API sin parámetros adicionales
        response = api.nba.games.list()
        games = response.get('data', [])
        all_games.extend(games)
    except Exception as e:
        print(f"Error al obtener datos de la API: {e}")
    
    return all_games

# Obtener los datos de la API
api_data = get_api_data()

# Si no hay datos, salir
if not api_data:
    print("No se obtuvieron datos de la API.")
    exit()

# Convertir los datos de la API en un DataFrame
df_api = pd.DataFrame(api_data)

# ---- 1. LIMPIEZA DEL DATASET DE LA API ----

# Eliminar duplicados
df_api = df_api.drop_duplicates()

# Filtrar solo las columnas necesarias, por ejemplo:
relevant_columns = ['id', 'home_team', 'visitor_team', 'score', 'date', 'season']
df_api = df_api[relevant_columns]

# Convertir la columna de fecha a formato datetime
df_api['date'] = pd.to_datetime(df_api['date'], errors='coerce')

# Rellenar valores nulos (si es apropiado)
df_api = df_api.fillna(0)

# ---- 2. GUARDAR EL DATASET LIMPIO ----

# Crear un nombre de archivo con la fecha actual
today = datetime.now().strftime("%Y-%m-%d")
cleaned_api_csv_path = os.path.join(RAW_DIR, f"{today}_clean_api.csv")

# Guardar el dataset limpio
df_api.to_csv(cleaned_api_csv_path, index=False)

# Resumen de cambios realizados
print(f"✅ Dataset de la API limpio y guardado en: {cleaned_api_csv_path}")
print("\nResumen de limpieza:")
print(f"- Número de registros obtenidos de la API: {len(api_data)}")
print(f"- Número de registros después de eliminar duplicados: {df_api.duplicated().sum()}")
print(f"- Número de valores nulos después de limpieza: {df_api.isnull().sum().sum()}")
