import pandas as pd
import torch
import joblib
import os
from model_v2_pipeline import preparar_datos, NBA_Points_Model

# Paso 1: Cargar el CSV de estadísticas
DATA_DIR = "data/raw"
filename = os.listdir(DATA_DIR)[0]  
df = pd.read_csv(os.path.join(DATA_DIR, filename))

print("Columnas del dataframe:", df.columns.tolist())
print("Equipos disponibles:", df['Team'].unique())

# Paso 2: Filtrar solo los equipos que van a jugar hoy
equipos_objetivo = ['ORL', 'BOS', 'MIL', 'IND', 'MIN', 'LAL']
df_filtrado = df[df['Team'].isin(equipos_objetivo)].copy()

print(f"Cantidad de registros después del filtrado: {len(df_filtrado)}")

# Paso 3: Preparar datos
# Asegurarse que los datos están en orden correcto
X = df_filtrado.select_dtypes(include=['number'])  # Solo columnas numéricas

# Paso 4: Cargar scaler
scaler = joblib.load(os.path.join('models', 'scaler.pkl'))

print("Columnas que espera el scaler:", scaler.feature_names_in_)
X = X[scaler.feature_names_in_]  # Reordenar columnas

# Paso 5: Normalizar
X_scaled = scaler.transform(X)

# Paso 6: Convertir a tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Paso 7: Cargar el modelo
model = NBA_Points_Model(input_dim=X_tensor.shape[1])
model.load_state_dict(torch.load('models/modelo_deep_learning.pth'))
model.eval()

# Paso 8: Predicciones
with torch.no_grad():
    predictions = model(X_tensor).squeeze()

df_filtrado['Predicted_Points'] = predictions.numpy()

# Paso 9: Agrupar las predicciones por equipo
resultados_por_equipo = df_filtrado.groupby('Team')['Predicted_Points'].sum()

# Paso 10: Mostrar los resultados
print("\n🎯 Predicción de puntos totales por equipo para hoy:")
for equipo, puntos in resultados_por_equipo.items():
    print(f"{equipo}: {puntos:.2f} puntos")
