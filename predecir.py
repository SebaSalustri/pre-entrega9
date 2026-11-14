
import pandas as pd
import joblib

# Cargar pipeline entrenado
pipeline = joblib.load("pipeline_churn.joblib")

# Nuevo cliente
nuevo_cliente = pd.DataFrame({
    "edad": [29],
    "antiguedad_meses": [4],
    "gasto_mensual": [35],
    "plan": ["Basico"],
    "medio_pago": ["Tarjeta"]
})

# Realizar prediccion
prediccion = pipeline.predict(nuevo_cliente)[0]

# Obtener probabilidad de abandono
probabilidad = pipeline.predict_proba(
    nuevo_cliente
)[0][1]

print("PREDICCION PARA NUEVO CLIENTE")

print(f"Churn predicho: {prediccion}")

print(
    f"Probabilidad de abandono: "
    f"{probabilidad:.2%}"
)

if prediccion == 1:
    print("Cliente con riesgo de abandono.")
else:
    print("Cliente sin abandono predicho.")
