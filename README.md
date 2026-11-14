# Pipeline de Machine Learning para predicción de Churn

## Descripción

Proyecto de Machine Learning orientado a la predicción de abandono de clientes (Churn), utilizando Python y Scikit-learn.

El objetivo es implementar un pipeline que integre el preprocesamiento de datos y el entrenamiento de un modelo de clasificación, evitando la fuga de datos (data leakage) y garantizando la consistencia entre entrenamiento e inferencia.

## Tecnologías utilizadas

- Python
- Pandas
- Scikit-learn
- Joblib

## Dataset

Se utiliza un dataset sintético de 40 clientes con variables numéricas y categóricas.

### Variables numéricas

- edad
- antiguedad_meses
- gasto_mensual

### Variables categóricas

- plan
- medio_pago

### Variable objetivo

- churn: indica si el cliente abandonó el servicio.
  - 0: cliente que permanece.
  - 1: cliente que abandona.

La columna cliente_id se excluye de las variables predictoras porque corresponde únicamente a un identificador.

El dataset contiene valores faltantes introducidos intencionalmente para comprobar el funcionamiento de las técnicas de imputación.

## Exploración de datos

| Variable | Valores nulos |
|---|---:|
| cliente_id | 0 |
| edad | 1 |
| antiguedad_meses | 0 |
| gasto_mensual | 1 |
| plan | 1 |
| medio_pago | 1 |
| churn | 0 |

## Preprocesamiento

Se utiliza ColumnTransformer para aplicar diferentes transformaciones según el tipo de variable.

### Variables numéricas

- SimpleImputer con estrategia de mediana para completar valores faltantes.
- StandardScaler para estandarizar las variables numéricas.

### Variables categóricas

- SimpleImputer con estrategia most_frequent para completar valores faltantes.
- OneHotEncoder para transformar las categorías en variables numéricas.

Se utiliza handle_unknown='ignore' para permitir que el pipeline procese categorías desconocidas durante la inferencia.

## Arquitectura del pipeline

El pipeline está compuesto por dos etapas principales:

1. Preprocesamiento mediante ColumnTransformer.
2. Clasificación mediante LogisticRegression.

El preprocesador y el estimador se integran dentro de un único objeto Pipeline de Scikit-learn.

## Prevención de Data Leakage

La división de los datos se realiza antes de entrenar el pipeline mediante train_test_split.

Se utiliza una distribución del 80% para entrenamiento y del 20% para evaluación.

| Conjunto | Registros |
|---|---:|
| Entrenamiento | 32 |
| Evaluación | 8 |
| Total | 40 |

Se utiliza random_state=42 para garantizar la reproducibilidad y stratify=y para conservar la distribución de clases.

El pipeline se ajusta únicamente con los datos de entrenamiento mediante pipeline.fit(X_train, y_train).

De esta manera, los parámetros de imputación, escalado y codificación se aprenden exclusivamente del conjunto de entrenamiento.

El conjunto de evaluación se utiliza únicamente para medir el rendimiento del modelo.

## Modelo utilizado

Se utiliza LogisticRegression como estimador de clasificación binaria.

El modelo permite estimar la probabilidad de abandono de un cliente a partir de sus características.

## Resultados obtenidos

### Accuracy

El modelo obtuvo un Accuracy de 0.8750, equivalente al 87.5%.

Esto significa que clasificó correctamente 7 de los 8 registros del conjunto de evaluación.

### Reporte de clasificación

| Clase | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 1.00 | 0.75 | 0.86 | 4 |
| 1 | 0.80 | 1.00 | 0.89 | 4 |
| Macro avg | 0.90 | 0.88 | 0.87 | 8 |
| Weighted avg | 0.90 | 0.88 | 0.87 | 8 |

### Matriz de confusión

| Clase real / predicha | No Churn (0) | Churn (1) |
|---|---:|---:|
| No Churn (0) | 3 | 1 |
| Churn (1) | 0 | 4 |

El modelo identificó correctamente los cuatro clientes que abandonaron el servicio.

Se registró un falso positivo correspondiente a un cliente que permaneció, pero fue clasificado como cliente con riesgo de abandono.

## Persistencia del pipeline

Se utiliza Joblib para exportar el pipeline entrenado:

```python
joblib.dump(pipeline, "pipeline_churn.joblib")
```

El archivo generado contiene tanto el preprocesamiento como el modelo entrenado.

Esto permite reutilizar el pipeline sin necesidad de repetir el entrenamiento ni implementar nuevamente las transformaciones.

## Inferencia

El archivo predecir.py carga el pipeline entrenado mediante Joblib y realiza una predicción sobre un cliente nuevo.

### Datos del cliente

| Variable | Valor |
|---|---|
| edad | 29 |
| antiguedad_meses | 4 |
| gasto_mensual | 35 |
| plan | Basico |
| medio_pago | Tarjeta |

### Resultado obtenido

- Churn predicho: 1
- Probabilidad estimada de abandono: 92.74%
- Clasificación: cliente con riesgo de abandono.

El pipeline aplica automáticamente las mismas transformaciones aprendidas durante el entrenamiento.

## Estructura del proyecto

```text
pipeline-churn-sklearn/
├── datos/
│   └── clientes.csv
├── entrenar_pipeline.py
├── predecir.py
├── requirements.txt
├── .gitignore
└── README.md
```

El archivo pipeline_churn.joblib se genera automáticamente al ejecutar el entrenamiento.

## Instalación

Instalar las dependencias desde la carpeta raíz del proyecto:

```bash
pip install -r requirements.txt
```

## Ejecución

### Entrenar y guardar el pipeline

```bash
python entrenar_pipeline.py
```

### Realizar una predicción

```bash
python predecir.py
```

Es necesario ejecutar primero el entrenamiento para generar el archivo pipeline_churn.joblib.

## Rendimiento

El tiempo total del proceso de entrenamiento se midió utilizando time.perf_counter().

| Operación | Tiempo de ejecución (segundos) |
|---|---:|
| Entrenamiento, evaluación y persistencia | 0.255133 |

El tiempo incluye la carga de datos, la división de los conjuntos, el entrenamiento del pipeline, la evaluación y la exportación del modelo.

Los resultados corresponden a una ejecución realizada en el equipo de desarrollo y pueden variar según el hardware y las condiciones de ejecución.

## Limitaciones

El dataset utilizado es sintético y contiene únicamente 40 registros.

Por este motivo, las métricas obtenidas tienen un propósito educativo y no permiten garantizar el rendimiento del modelo sobre clientes reales.

La probabilidad de abandono obtenida durante la inferencia es una estimación del modelo y no representa una probabilidad validada en un entorno de producción.

## Conclusión

El uso de Pipeline y ColumnTransformer permite encapsular el preprocesamiento y el entrenamiento en un único objeto reutilizable.

Esta arquitectura facilita la prevención de data leakage, garantiza la consistencia de las transformaciones y permite automatizar el entrenamiento y la inferencia.

La persistencia mediante Joblib permite reutilizar el modelo entrenado en futuras aplicaciones de Machine Learning.
