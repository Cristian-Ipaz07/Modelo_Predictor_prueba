from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import os

# Cargar el CSV original
DATA_DIR = "data/raw"
filename = os.listdir(DATA_DIR)[0]
df = pd.read_csv(os.path.join(DATA_DIR, filename))

# Preparar datos usando tu función
from model_pipeline import preparar_datos
team_features = preparar_datos(df)

# Entrenar el nuevo scaler
scaler = StandardScaler()
scaler.fit(team_features)

# Guardarlo
joblib.dump(scaler, os.path.join('models', 'scaler.pkl'))

print("✅ Nuevo scaler guardado correctamente.")
