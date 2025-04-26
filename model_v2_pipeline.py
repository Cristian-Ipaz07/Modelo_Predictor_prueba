import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


sns.set(style="whitegrid")

DATA_DIR = "data/raw"
filename = os.listdir(DATA_DIR)[0]  
df = pd.read_csv(os.path.join(DATA_DIR, filename))

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

print("✅ Dataset cargado correctamente.")
print("Número de filas:", len(df))
print("Columnas:", df.columns.tolist())

# Agrupar por equipo y fecha → obtener stats agregadas por partido y equipo
team_game_stats = df.groupby(['Date', 'Team']).agg({
    'PTS': 'sum',     # Total de puntos por equipo
    'AST': 'sum',     # Asistencias totales
    'TRB': 'sum',     # Rebotes totales
    'TOV': 'sum',     # Pérdidas
    'FG%': 'mean',    # Promedio de efectividad en tiros
    '3P%': 'mean',    # Promedio de triples
    'FT%': 'mean',    # Promedio de tiros libres
    '+/-': 'sum',     # +/- total del equipo
}).reset_index()

# Renombrar para claridad
team_game_stats.rename(columns={
    'PTS': 'Team_PTS',
    'AST': 'Team_AST',
    'TRB': 'Team_REB',
    'TOV': 'Team_TOV',
    'FG%': 'Team_FG%',
    '3P%': 'Team_3P%',
    'FT%': 'Team_FT%',
    '+/-': 'Team_PlusMinus'
}, inplace=True)

# Unir tabla consigo misma para obtener local vs visitante por fecha
# Suponemos que en cada fecha hay exactamente dos equipos por juego
games = team_game_stats.merge(team_game_stats, on='Date')
games = games[games['Team_x'] != games['Team_y']]  # Evitar unión con uno mismo

# Eliminar duplicados espejo (mismo partido dos veces en distinto orden)
games['key'] = games.apply(lambda x: '-'.join(sorted([x['Team_x'], x['Team_y']])), axis=1)
games = games.drop_duplicates(subset=['Date', 'key']).drop(columns='key')

# Crear variables target y auxiliares
games['Total_Points'] = games['Team_PTS_x'] + games['Team_PTS_y']
games['Team_A'] = games['Team_x']
games['Team_B'] = games['Team_y']

# Mantener columnas necesarias
games_df = games[[
    'Date', 'Team_A', 'Team_B',
    'Team_PTS_x', 'Team_PTS_y',
    'Team_AST_x', 'Team_REB_x', 'Team_TOV_x', 'Team_FG%_x', 'Team_3P%_x', 'Team_FT%_x', 'Team_PlusMinus_x',
    'Team_AST_y', 'Team_REB_y', 'Team_TOV_y', 'Team_FG%_y', 'Team_3P%_y', 'Team_FT%_y', 'Team_PlusMinus_y',
    'Total_Points'
]]

print("✅ Tabla de partidos creada.")
print("Número de partidos:", len(games_df))

# Ordenar por equipo y fecha para calcular EMA correctamente
team_game_stats = team_game_stats.sort_values(by=['Team', 'Date'])

# EMA de los últimos 5 y 10 partidos por equipo
team_game_stats['PTS_EMA_5'] = team_game_stats.groupby('Team')['Team_PTS'].transform(lambda x: x.ewm(span=5).mean())
team_game_stats['PTS_EMA_10'] = team_game_stats.groupby('Team')['Team_PTS'].transform(lambda x: x.ewm(span=10).mean())

# Calcular días de descanso
team_game_stats['Days_Rest'] = team_game_stats.groupby('Team')['Date'].diff().dt.days.fillna(0)

# Usamos el df original porque 'Result' se borró en el agrupado

# Creamos diccionario de localía por equipo y fecha
df['Home'] = df['Result'].apply(lambda x: 1 if isinstance(x, str) and x.startswith('W') else 0)
home_map = df.groupby(['Date', 'Team'])['Home'].mean().round().astype(int)

# Fusionar con team_game_stats
team_game_stats = team_game_stats.merge(home_map, on=['Date', 'Team'], how='left')

# Renombramos para distinguir equipos
team_feats = team_game_stats[['Date', 'Team', 'Team_PTS', 'PTS_EMA_5', 'PTS_EMA_10', 'Days_Rest', 'Home']]

# Merge para Team A
games_df = games_df.merge(team_feats, left_on=['Date', 'Team_A'], right_on=['Date', 'Team'], how='left')
games_df.rename(columns={
    'Team_PTS': 'PTS_A',
    'PTS_EMA_5': 'PTS_EMA_5_A', 'PTS_EMA_10': 'PTS_EMA_10_A',
    'Days_Rest': 'Days_Rest_A', 'Home': 'Home_A'
}, inplace=True)
games_df.drop(columns=['Team'], axis=1, inplace=True)

# Merge para Team B
games_df = games_df.merge(team_feats, left_on=['Date', 'Team_B'], right_on=['Date', 'Team'], how='left')
games_df.rename(columns={
    'Team_PTS': 'PTS_B',
    'PTS_EMA_5': 'PTS_EMA_5_B', 'PTS_EMA_10': 'PTS_EMA_10_B',
    'Days_Rest': 'Days_Rest_B', 'Home': 'Home_B'
}, inplace=True)
games_df.drop(columns=['Team'], axis=1, inplace=True)


"""


# EMA
sns.histplot(games_df['PTS_EMA_5_A'], kde=True)
plt.title("Distribución EMA 5 partidos (Team A)")
plt.show()

# Días de descanso
sns.histplot(games_df['Days_Rest_A'], kde=True)
plt.title("Distribución días de descanso (Team A)")
plt.show()

# Localía
sns.countplot(x='Home_A', data=games_df)
plt.title("Distribución de localía (Team A)")
plt.show()


print(games_df.isnull().sum())


# Histograma de puntos totales por partido
plt.figure(figsize=(10, 6))
sns.histplot(games_df['PTS_A'] + games_df['PTS_B'], kde=True)
plt.title("Distribución de puntos totales por partido")
plt.xlabel("Puntos Totales")
plt.ylabel("Frecuencia")
plt.show()

# Scatter plots
plt.figure(figsize=(10, 6))

# EMA vs Puntos Totales
sns.scatterplot(x=games_df['PTS_EMA_5_A'] + games_df['PTS_EMA_5_B'], y=games_df['PTS_A'] + games_df['PTS_B'])
plt.title("EMA 5 partidos vs Puntos Totales")
plt.xlabel("EMA 5 partidos")
plt.ylabel("Puntos Totales")
plt.show()

# Días de descanso vs Puntos Totales
plt.figure(figsize=(10, 6))
sns.scatterplot(x=games_df['Days_Rest_A'] + games_df['Days_Rest_B'], y=games_df['PTS_A'] + games_df['PTS_B'])
plt.title("Días de descanso vs Puntos Totales")
plt.xlabel("Días de descanso")
plt.ylabel("Puntos Totales")
plt.show()

# Matriz de correlación
corr_matrix = games_df.select_dtypes(include=['number']).corr()


plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Matriz de correlación")
plt.show()

# Revisión de outliers con boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x=games_df['PTS_A'] + games_df['PTS_B'])
plt.title("Revisión de outliers en puntos totales")
plt.show()

# Revisión de balance de clases (si es necesario)
sns.countplot(x='Home_A', data=games_df)
plt.title("Distribución de partidos como local")
plt.show()

"""

## Modelos Baseline (Machine Learning Clásico)

# Objetivo
games_df['PTS_TOTAL'] = games_df['PTS_A'] + games_df['PTS_B']

# Features numéricas (puedes añadir/quitar según preferencia)
features = [
    'PTS_EMA_5_A', 'PTS_EMA_10_A',
    'PTS_EMA_5_B', 'PTS_EMA_10_B',
    'Days_Rest_A', 'Days_Rest_B',
    'Home_A', 'Home_B'
]

X = games_df[features]
y = games_df['PTS_TOTAL']

# División en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelos
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
}

# Métricas
results = {}


for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    results[name] = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": rmse,
        "R2": r2_score(y_test, preds)
    }

# Mostrar resultados
results_df = pd.DataFrame(results).T
print("📊 Resultados Baseline:")
print(results_df)


# Ejemplo: XGBoost
best_model = models['XGBoost']
y_pred = best_model.predict(X_test)
"""
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Puntos reales")
plt.ylabel("Puntos predichos")
plt.title("XGBoost: Predicción vs Real")
plt.grid(True)
plt.show()
"""
# modelo PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset



def preparar_datos(df):
    """
    Función para preparar las estadísticas de jugadores y calcular
    las características necesarias para la predicción de puntos por equipo.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    hoy = pd.Timestamp.today()
    df = df[df['Date'] < hoy]  # Solo partidos jugados

    # Ordenar y quedarnos con los últimos 10 partidos por jugador
    df = df.sort_values(['Player', 'Date'], ascending=[True, False])
    df = df.groupby('Player').head(10)

    # Calcular EMAs de puntos (5 y 10 partidos)
    df['PTS_EMA_5'] = df.groupby('Player')['PTS'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    df['PTS_EMA_10'] = df.groupby('Player')['PTS'].transform(lambda x: x.ewm(span=10, adjust=False).mean())

    # Calcular Days Rest (descanso entre partidos)
    df['Days_Rest'] = df.groupby('Player')['Date'].diff().dt.days.fillna(0)

    # Identificar si el partido es en casa o fuera
    if 'Home/Away' in df.columns:
        df['Home'] = df['Home/Away'].apply(lambda x: 1 if x == 'Home' else 0)
    else:
        # Si no existe la columna, asumimos todo como 0 (neutro)
        df['Home'] = 0

    # Promediar estadísticas por equipo y fecha
    team_game_stats = df.groupby(['Team', 'Date']).agg({
        'PTS': 'sum',
        'PTS_EMA_5': 'mean',
        'PTS_EMA_10': 'mean',
        'Days_Rest': 'mean',
        'Home': 'mean'
    }).reset_index()

    # Ordenar por equipo y fecha para calcular bien EMAs
    team_game_stats = team_game_stats.sort_values(['Team', 'Date'])

    # Volver a calcular EMAs pero a nivel equipo
    team_game_stats['PTS_EMA_5'] = team_game_stats.groupby('Team')['PTS'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    team_game_stats['PTS_EMA_10'] = team_game_stats.groupby('Team')['PTS'].transform(lambda x: x.ewm(span=10, adjust=False).mean())

    # Calcular días de descanso entre partidos de equipos
    team_game_stats['Days_Rest'] = team_game_stats.groupby('Team')['Date'].diff().dt.days.fillna(0)

    # --- Generar emparejamientos de partidos ---

    games = team_game_stats.merge(team_game_stats, on='Date')
    games = games[games['Team_x'] != games['Team_y']]  # Evitar que se junte consigo mismo

    # Eliminar duplicados espejo
    games['key'] = games.apply(lambda x: '-'.join(sorted([x['Team_x'], x['Team_y']])), axis=1)
    games = games.drop_duplicates(subset=['Date', 'key']).drop(columns='key')

    # Crear variables finales
    games['PTS_EMA_5_A'] = games['PTS_EMA_5_x']
    games['PTS_EMA_10_A'] = games['PTS_EMA_10_x']
    games['Days_Rest_A'] = games['Days_Rest_x']
    games['Home_A'] = games['Home_x']

    games['PTS_EMA_5_B'] = games['PTS_EMA_5_y']
    games['PTS_EMA_10_B'] = games['PTS_EMA_10_y']
    games['Days_Rest_B'] = games['Days_Rest_y']
    games['Home_B'] = games['Home_y']

    # Devolver solo las columnas necesarias
    features = games[['PTS_EMA_5_A', 'PTS_EMA_10_A', 'PTS_EMA_5_B', 'PTS_EMA_10_B',
                      'Days_Rest_A', 'Days_Rest_B', 'Home_A', 'Home_B']]

    return features



#definir arquitectura

class NBA_Points_Model(nn.Module):
    def __init__(self, input_dim):
        super(NBA_Points_Model, self).__init__()
        self.dense1 = nn.Linear(input_dim, 128)  # 🔥 aquí ponemos 8
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.output(x)
        return x



def train_model():
    print("Entrenando el modelo...")

    # Cargar y preprocesar los datos
    DATA_DIR = "data/raw"
    filename = os.listdir(DATA_DIR)[0]
    df = pd.read_csv(os.path.join(DATA_DIR, filename))

    team_features = preparar_datos(df)  # Usa tu función preparar_datos actualizada
    X = team_features[['PTS_EMA_5_A', 'PTS_EMA_10_A', 'PTS_EMA_5_B', 'PTS_EMA_10_B',
                       'Days_Rest_A', 'Days_Rest_B', 'Home_A', 'Home_B']].values
    y = team_features['PTS_EMA_5_A'].values
    
    # Normalizar las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    import joblib
    model_dir = 'models/'
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    print("✅ Scaler guardado como scaler.pkl")

    # Convertir a tensores de PyTorch
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    X_train_tensor = X_train
    X_val_tensor = X_val
    y_train_tensor = y_train
    y_val_tensor = y_val

    # Crear un DataLoader para el entrenamiento
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


    # Inicializar el modelo
    input_dim = X.shape[1]  # Número de características de entrada
    model = NBA_Points_Model(input_dim)

    # Definir el optimizador y la función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Número de epochs
    # Early stopping
    patience = 15
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    epochs = 300

    # Entrenamiento
    for epoch in range(epochs):
        model.train()  # Establecer el modelo en modo de entrenamiento
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calcular la pérdida
            loss = criterion(outputs, targets)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Promedio de la pérdida para esta época
        avg_train_loss = running_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("✅ Entrenamiento completo y modelo guardado.")

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    # Modo evaluación
    # Predicciones finales
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).numpy()
        y_true = y_test_tensor.numpy()

    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)

    print(f"📈 MAE: {mae:.2f}")
    print(f"📈 RMSE: {rmse:.2f}")
    print(f"📈 R2 Score: {r2:.2f}")

    torch.save(model.state_dict(), "models/modelo_deep_learning.pth")

 
    # Gráfica
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, preds, alpha=0.5)
    plt.xlabel("Actual Points")
    plt.ylabel("Predicted Points")
    plt.title("Predicción vs Real")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')  # línea de igualdad
    plt.show()

if __name__ == '__main__':
    print("Ejecutando modelo_pipeline.py directamente...")
    train_model()

