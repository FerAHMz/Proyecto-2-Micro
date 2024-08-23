import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Carga de datos
data = pd.read_csv('Almond.csv')

# Visualización inicial de datos
print(data.head())
print(data.describe().T)
print(data.isnull().sum())
print(data.duplicated().sum())
print("Shape of data:", data.shape)

# Limpieza de datos
data = data.drop(columns=['Unnamed: 0'])
data.drop_duplicates(inplace=True)  # Eliminar duplicados si existen

# Conteo de tipos
type_counts = data['Type'].value_counts()
print(type_counts)

# Visualización de distribución de tipos
fig = px.bar(
    type_counts,
    x=type_counts.index,
    y=type_counts.values,
    labels={'x': 'Type', 'y': 'Count'},
    title='Type Distribution'
)
fig.update_layout(width=500, height=500)
fig.show()

# Imputación de valores faltantes para todas las columnas numéricas con NaN
columns_with_na = ['Length (major axis)', 'Width (minor axis)', 'Thickness (depth)', 'Roundness', 'Aspect Ratio', 'Eccentricity']
imputer = KNNImputer(n_neighbors=5)
data[columns_with_na] = imputer.fit_transform(data[columns_with_na])

# Visualización de histogramas para verificar distribuciones después de la imputación
fig, axs = plt.subplots(len(columns_with_na), 1, figsize=(8, 4 * len(columns_with_na)))
for i, column in enumerate(columns_with_na):
    sns.histplot(data[column], ax=axs[i], kde=False)
    axs[i].set_title(f'Histogram of {column}')
    axs[i].set_xlabel(column)
    axs[i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

# Codificación de etiquetas
label_encoder = LabelEncoder()
data['Type'] = label_encoder.fit_transform(data['Type'])

# División del conjunto de datos
X = data.drop('Type', axis=1)
y = data['Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

# Modelos de clasificación
models = {
    'Random Forest': RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    print(f"{name} Classifier")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print()

# Comparación de predicciones
results_df = pd.DataFrame({
    'Original Values': y_test,
    'RF Predictions': models['Random Forest'].predict(X_test),
    'XGB Predictions': models['XGBoost'].predict(X_test),
    'LGB Predictions': models['LightGBM'].predict(X_test)
})
print("Comparison of Model Predictions and Original Values:")
print(results_df.head())
