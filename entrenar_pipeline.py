
import pandas as pd
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Inicio del proceso
inicio = time.perf_counter()

# ==========================
# 1. CARGA DE DATOS
# ==========================

df = pd.read_csv("datos/clientes.csv")

print("INFORMACION DEL DATASET")
print(f"Cantidad de registros: {len(df)}")

print("\nValores nulos por columna:")
print(df.isnull().sum())

# ==========================
# 2. DEFINICION DE VARIABLES
# ==========================

# Variables predictoras
X = df.drop(columns=["cliente_id", "churn"])

# Variable objetivo
y = df["churn"]

# Columnas numericas
columnas_numericas = [
    "edad",
    "antiguedad_meses",
    "gasto_mensual"
]

# Columnas categoricas
columnas_categoricas = [
    "plan",
    "medio_pago"
]

# ==========================
# 3. DIVISION TRAIN / TEST
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDIVISION DE DATOS")
print(f"Entrenamiento: {len(X_train)} registros")
print(f"Evaluacion: {len(X_test)} registros")

# ==========================
# 4. TRANSFORMADORES
# ==========================

# Preprocesamiento numerico
pipeline_numerico = Pipeline(steps=[
    (
        "imputacion",
        SimpleImputer(strategy="median")
    ),
    (
        "escalado",
        StandardScaler()
    )
])

# Preprocesamiento categorico
pipeline_categorico = Pipeline(steps=[
    (
        "imputacion",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "codificacion",
        OneHotEncoder(handle_unknown="ignore")
    )
])

# Combinar transformaciones
preprocesador = ColumnTransformer(
    transformers=[
        (
            "numericas",
            pipeline_numerico,
            columnas_numericas
        ),
        (
            "categoricas",
            pipeline_categorico,
            columnas_categoricas
        )
    ]
)

# ==========================
# 5. PIPELINE COMPLETO
# ==========================

pipeline = Pipeline(steps=[
    ("preprocesamiento", preprocesador),
    (
        "modelo",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# ==========================
# 6. ENTRENAMIENTO
# ==========================

print("\nEntrenando pipeline...")

pipeline.fit(X_train, y_train)

# ==========================
# 7. EVALUACION
# ==========================

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nRESULTADOS DEL MODELO")

print(f"Accuracy: {accuracy:.4f}")

print("\nReporte de clasificacion:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nMatriz de confusion:")
print(confusion_matrix(y_test, y_pred))

# ==========================
# 8. PERSISTENCIA
# ==========================

joblib.dump(
    pipeline,
    "pipeline_churn.joblib"
)

print("\nPipeline guardado correctamente.")

# ==========================
# 9. TIEMPO TOTAL
# ==========================

tiempo_total = time.perf_counter() - inicio

print(f"Tiempo total: {tiempo_total:.6f} segundos")
