# Students Social Media & Academic Performance Analysis Backend

Backend completo en Flask/Python para análisis de redes sociales y rendimiento académico con Machine Learning.

## Características

- **Análisis de Machine Learning**: 6 algoritmos implementados (Regresión Linear/Logística, Árboles de Decisión, Random Forest, SVM, K-Means)
- **API RESTful completa**: Endpoints para usuarios, modelos ML, análisis, predicciones y visualizaciones
- **Almacenamiento en Excel**: Sistema robusto de archivos Excel con backups automáticos
- **Visualizaciones avanzadas**: Gráficos interactivos y dashboard de análisis
- **Validaciones robustas**: Sistema completo de validación de datos
- **CORS configurado**: Listo para integración con frontend React+TypeScript

## Estructura del Proyecto

```
├── app.py                  # Aplicación principal Flask
├── config.py              # Configuración global
├── requirements.txt       # Dependencias Python
├── data/
│   ├── excel/             # Archivos Excel de datos
│   │   ├── users.xlsx
│   │   ├── ml_models.xlsx
│   │   └── predictions.xlsx
│   └── backups/           # Backups automáticos
├── models/
│   └── data_models.py     # Modelos de datos (dataclasses)
├── routes/
│   ├── users.py           # Endpoints de usuarios
│   ├── models.py          # Endpoints de modelos ML
│   ├── analysis.py        # Endpoints de análisis
│   └── visualizations.py  # Endpoints de visualizaciones
├── services/
│   ├── ml_service.py      # Servicios de Machine Learning
│   ├── excel_service.py   # Servicios de Excel
│   └── visualization_service.py # Servicios de visualización
├── utils/
│   ├── excel_utils.py     # Utilidades para Excel
│   ├── validators.py      # Validaciones
│   └── helpers.py         # Funciones auxiliares
└── static/
    └── plots/             # Gráficos generados
```

## Instalación y Configuración

### Requisitos Previos

- Python 3.8+
- pip

### Pasos de Instalación

1. **Clonar el repositorio**

   ```bash
   git clone "URL_REPOSITORIO"
   cd students-ED
   ```

2. **Crear entorno virtual**

   ```bash
   python -m venv venv
   ```

3. **Activar entorno virtual**

   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

4. **Instalar dependencias**

   ```bash
   pip install -r requirements.txt
   ```

5. **Ejecutar la aplicación**
   ```bash
   python app.py
   ```

La aplicación estará disponible en `http://localhost:5000`

## API Endpoints

### Usuarios (`/api/users`)

#### GET /api/users

Obtener todos los usuarios

```json
{
  "users": [
    {
      "id": "user_001",
      "name": "Juan Pérez",
      "age": 20,
      "gender": "M",
      "country": "Mexico",
      "social_media_usage": 5.5,
      "academic_performance": 8.2
    }
  ]
}
```

#### POST /api/users

Crear nuevo usuario

```json
{
  "name": "María García",
  "age": 19,
  "gender": "F",
  "country": "Spain",
  "social_media_usage": 4.2,
  "academic_performance": 9.1
}
```

#### GET /api/users/{user_id}

Obtener usuario específico

#### PUT /api/users/{user_id}

Actualizar usuario

#### DELETE /api/users/{user_id}

Eliminar usuario

### Modelos ML (`/api/models`)

#### GET /api/models

Listar modelos entrenados

#### POST /api/models/train

Entrenar nuevo modelo

```json
{
  "name": "Modelo_Prediccion_2024",
  "algorithm": "linear_regression",
  "features": ["age", "social_media_usage"],
  "target": "academic_performance"
}
```

#### GET /api/models/{model_id}

Obtener detalles del modelo

#### DELETE /api/models/{model_id}

Eliminar modelo

### Análisis y Predicciones (`/api/analysis`)

#### POST /api/analysis/predict

Realizar predicción

```json
{
  "model_id": "model_001",
  "data": {
    "age": 20,
    "social_media_usage": 6.0
  }
}
```

#### GET /api/analysis/compare-algorithms

Comparar rendimiento de algoritmos

```json
{
  "features": ["age", "social_media_usage"],
  "target": "academic_performance"
}
```

#### POST /api/analysis/correlation

Análisis de correlación

```json
{
  "features": ["age", "social_media_usage", "academic_performance"]
}
```

### Visualizaciones (`/api/visualizations`)

#### POST /api/visualizations/histogram

Generar histograma

```json
{
  "column": "age",
  "title": "Distribución de Edades"
}
```

#### POST /api/visualizations/scatter

Generar gráfico de dispersión

```json
{
  "x_column": "social_media_usage",
  "y_column": "academic_performance",
  "title": "Relación Uso de Redes vs Rendimiento"
}
```

#### POST /api/visualizations/correlation-heatmap

Mapa de calor de correlaciones

#### GET /api/visualizations/dashboard

Dashboard completo con múltiples visualizaciones

## Algoritmos de Machine Learning

### Algoritmos Implementados

1. **Regresión Lineal**: Predicción continua de rendimiento académico
2. **Regresión Logística**: Clasificación binaria de alto/bajo rendimiento
3. **Árboles de Decisión**: Análisis interpretable de factores
4. **Random Forest**: Ensemble para mayor precisión
5. **SVM**: Support Vector Machine para clasificación/regresión
6. **K-Means**: Clustering de perfiles de estudiantes

### Métricas de Evaluación

- **Regresión**: MAE, MSE, RMSE, R²
- **Clasificación**: Accuracy, Precision, Recall, F1-Score
- **Clustering**: Silhouette Score, Inertia

## Ejemplos de Uso

### Crear Usuario y Entrenar Modelo

```python
import requests

# Crear usuario
user_data = {
    "name": "Ana López",
    "age": 21,
    "gender": "F",
    "country": "Argentina",
    "social_media_usage": 3.8,
    "academic_performance": 8.9
}

response = requests.post('http://localhost:5000/api/users', json=user_data)
print(response.json())

# Entrenar modelo
model_data = {
    "name": "Predictor_Rendimiento_2024",
    "algorithm": "random_forest",
    "features": ["age", "social_media_usage"],
    "target": "academic_performance"
}

response = requests.post('http://localhost:5000/api/models/train', json=model_data)
print(response.json())
```

### Realizar Predicción

```python
# Hacer predicción
prediction_data = {
    "model_id": "model_001",
    "data": {
        "age": 19,
        "social_media_usage": 7.2
    }
}

response = requests.post('http://localhost:5000/api/analysis/predict', json=prediction_data)
print(f"Predicción: {response.json()['prediction']}")
```

## Sistema de Archivos Excel

### Estructura de Datos

#### users.xlsx

- id, name, age, gender, country, social_media_usage, academic_performance

#### ml_models.xlsx

- id, name, algorithm, features, target, accuracy, created_at, file_path

#### predictions.xlsx

- id, model_id, input_data, prediction, confidence, created_at

### Backups Automáticos

El sistema crea backups automáticos antes de cada modificación:

- Ubicación: `data/backups/`
- Formato: `{filename}_{timestamp}.xlsx`
- Retención: Configurable en `config.py`

## Desarrollo y Testing

### Ejecutar en Modo Debug

```bash
export FLASK_ENV=development  # Linux/macOS
set FLASK_ENV=development     # Windows
python app.py
```

### Testing Manual de Endpoints

```bash
# Obtener usuarios
curl http://localhost:5000/api/users

# Crear usuario
curl -X POST http://localhost:5000/api/users \
  -H "Content-Type: application/json" \
  -d '{"name":"Test User","age":20,"gender":"M","country":"Test","social_media_usage":5.0,"academic_performance":7.5}'

# Entrenar modelo
curl -X POST http://localhost:5000/api/models/train \
  -H "Content-Type: application/json" \
  -d '{"name":"Test Model","algorithm":"linear_regression","features":["age","social_media_usage"],"target":"academic_performance"}'
```

## Integración con Frontend

### CORS Configurado

El backend está configurado para aceptar requests desde:

- `http://localhost:3000` (React dev server)
- `http://localhost:5173` (Vite dev server)
- Otros orígenes configurables en `config.py`

### Headers Recomendados

```javascript
const headers = {
  "Content-Type": "application/json",
  Accept: "application/json",
};

// Ejemplo con fetch
const response = await fetch("http://localhost:5000/api/users", {
  method: "GET",
  headers: headers,
});
```

## Configuración

### Variables de Entorno (config.py)

```python
# Rutas de archivos
DATA_DIR = 'data'
EXCEL_DIR = 'data/excel'
BACKUP_DIR = 'data/backups'
STATIC_DIR = 'static'
PLOTS_DIR = 'static/plots'

# Configuración de servidor
HOST = '0.0.0.0'
PORT = 5000
DEBUG = True

# CORS
CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:5173']
```

## Resolución de Problemas

### Error: "No module named 'pandas'"

```bash
pip install -r requirements.txt
```

### Error: "Permission denied" en archivos Excel

- Cerrar archivos Excel abiertos
- Verificar permisos de escritura en directorio `data/`

### Error: "FileNotFoundError"

- Los archivos Excel se crean automáticamente al iniciar
- Verificar que existe el directorio `data/excel/`

### Puerto 5000 ocupado

```bash
# Cambiar puerto en config.py o usar variable de entorno
export PORT=5001
python app.py
```

## Dependencias Principales

- **Flask**: Framework web
- **pandas**: Manipulación de datos
- **scikit-learn**: Machine Learning
- **matplotlib/seaborn**: Visualizaciones
- **openpyxl**: Manejo de archivos Excel
- **flask-cors**: CORS para frontend

## Contribución

1. Fork del repositorio
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT.
