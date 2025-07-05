# 📡 API ENDPOINTS - DOCUMENTACIÓN PARA FRONTEND

## 🌐 URL Base

```
http://localhost:5000
```

## 🔧 Headers Requeridos

```javascript
{
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}
```

---

## 👥 USUARIOS - `/api/users`

### 1. **GET** `/api/users` - Obtener todos los usuarios

**Response:**

```typescript
{
  users: Array<{
    id: number;
    name: string;
    email: string;
    age: number;
    gender: "Masculino" | "Femenino" | "Otro";
    education_level: "Bachillerato" | "Universidad" | "Posgrado";
    social_media_usage: number; // 1-10 horas
    academic_performance: number; // 0-100
    main_platform:
      | "Instagram"
      | "TikTok"
      | "Facebook"
      | "Twitter"
      | "YouTube"
      | "LinkedIn";
    study_hours: number;
    created_at: string;
  }>;
  total: number;
}
```

### 2. **POST** `/api/users` - Crear nuevo usuario

**Request Body:**

```typescript
{
  name: string; // Requerido
  email: string; // Requerido
  age: number; // Requerido
  gender: "Masculino" | "Femenino" | "Otro"; // Requerido
  education_level: "Bachillerato" | "Universidad" | "Posgrado"; // Requerido
  social_media_usage?: number; // Opcional, default: 5
  academic_performance?: number; // Opcional, default: 75.0
  main_platform?: string; // Opcional, default: "Instagram"
  study_hours?: number; // Opcional, default: 25
}
```

**Response (201):**

```typescript
{
  message: string;
  user: {
    id: number;
    name: string;
    email: string;
    age: number;
    gender: string;
    education_level: string;
    social_media_usage: number;
    academic_performance: number;
    main_platform: string;
    study_hours: number;
    created_at: string;
  }
}
```

### 3. **GET** `/api/users/{user_id}` - Obtener usuario específico

**Response (200):**

```typescript
{
  user: {
    id: number;
    name: string;
    email: string;
    age: number;
    gender: string;
    education_level: string;
    social_media_usage: number;
    academic_performance: number;
    main_platform: string;
    study_hours: number;
    created_at: string;
  }
}
```

### 4. **PUT** `/api/users/{user_id}` - Actualizar usuario

**Request Body:** (Campos opcionales para actualizar)

```typescript
{
  name?: string;
  email?: string;
  age?: number;
  gender?: "Masculino" | "Femenino" | "Otro";
  education_level?: "Bachillerato" | "Universidad" | "Posgrado";
  social_media_usage?: number;
  academic_performance?: number;
  main_platform?: string;
  study_hours?: number;
}
```

### 5. **DELETE** `/api/users/{user_id}` - Eliminar usuario

**Response (200):**

```typescript
{
  message: string;
}
```

---

## 🤖 MODELOS ML - `/api/models`

### 1. **GET** `/api/models` - Obtener todos los modelos

**Response:**

```typescript
{
  models: Array<{
    id: number;
    name: string;
    algorithm: string;
    description: string;
    accuracy: number;
    use_cases: string;
    use_cases_list: string[];
    is_locked: boolean;
    unlock_condition: string;
    created_at: string;
  }>;
  total: number;
}
```

### 2. **GET** `/api/models/{model_id}` - Obtener modelo específico

**Response:**

```typescript
{
  model: {
    id: number;
    name: string;
    algorithm: string;
    description: string;
    accuracy: number;
    use_cases: string;
    use_cases_list: string[];
    is_locked: boolean;
    unlock_condition: string;
    created_at: string;
  }
}
```

### 3. **POST** `/api/models/{model_id}/train` - Entrenar modelo con usuario

**Request Body:**

```typescript
{
  user_id: number; // Requerido
}
```

**Response (200):**

```typescript
{
  status: "success" | "error";
  model_id: string;
  user_id: string;
  timestamp: string;
  results: {
    accuracy?: number;
    prediction?: number;
    cluster?: number;
    risk_level?: string;
    interpretation: {
      summary: string;
      key_factors: string[];
      recommendations: string[];
      confidence_level: string;
    };
  };
  execution_time?: string;
  error?: string;
}
```

---

## 📊 ANÁLISIS - `/api`

### 1. **POST** `/api/analyze/user/{user_id}` - Análisis completo de usuario

**Response:**

```typescript
{
  user_info: {
    id: number;
    name: string;
    email: string;
    age: number;
    gender: string;
    education_level: string;
    social_media_usage: number;
    academic_performance: number;
    main_platform: string;
    study_hours: number;
  };
  comprehensive_analysis: {
    status: string;
    timestamp: string;
    user_profile: {
      risk_level: "Bajo" | "Medio" | "Alto";
      academic_prediction: number;
      social_media_impact: string;
      study_efficiency: string;
    };
    model_comparison: Array<{
      algorithm: string;
      accuracy: number;
      prediction: number;
      confidence: number;
    }>;
    interpretation: {
      summary: string;
      key_factors: string[];
      recommendations: string[];
    };
  };
  individual_models: {
    linear_regression: ModelResult;
    logistic_regression: ModelResult;
    kmeans_clustering: ModelResult;
    random_forest: ModelResult;
    decision_tree: ModelResult;
    support_vector_machine: ModelResult;
  };
  summary: {
    total_models_executed: number;
    models_with_errors: number;
    analysis_timestamp: string;
    risk_assessment: string;
    academic_prediction: number;
  };
}
```

### 2. **GET** `/api/predictions/{user_id}` - Obtener predicciones de usuario

**Response:**

```typescript
{
  user_id: number;
  predictions: Array<{
    id: number;
    model_id: number;
    model_name: string;
    prediction_value: number;
    confidence: number;
    risk_level: string;
    timestamp: string;
    input_data: object;
  }>;
  total: number;
  latest_prediction: {
    value: number;
    confidence: number;
    model_used: string;
    timestamp: string;
  }
}
```

### 3. **GET** `/api/analyze/models/compare` - Comparar rendimiento de modelos

**Query Parameters:**

```
?user_id=1 (opcional)
```

**Response:**

```typescript
{
  comparison_results: Array<{
    model_name: string;
    algorithm: string;
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    execution_time: string;
  }>;
  best_model: {
    name: string;
    accuracy: number;
    recommendation: string;
  }
  timestamp: string;
}
```

---

## 📈 VISUALIZACIONES - `/api/visualizations`

### 1. **GET** `/api/visualizations/dashboard` - Dashboard completo

**Response:**

```typescript
{
  visualizations: Array<{
    type: string;
    title: string;
    data: object;
    image_path?: string;
    insights: string[];
  }>;
  summary: {
    total_users: number;
    average_performance: number;
    most_used_platform: string;
    risk_distribution: {
      low: number;
      medium: number;
      high: number;
    };
  };
  recommendations: string[];
  timestamp: string;
}
```

### 2. **GET** `/api/visualizations/age-distribution` - Distribución de edades

**Response:**

```typescript
{
  type: "histogram";
  title: string;
  data: {
    ages: number[];
    frequencies: number[];
    bins: number[];
  };
  statistics: {
    mean_age: number;
    median_age: number;
    age_range: [number, number];
  };
  insights: string[];
  image_path?: string;
}
```

### 3. **GET** `/api/visualizations/social-vs-performance` - Correlación uso vs rendimiento

**Response:**

```typescript
{
  type: "scatter";
  title: string;
  data: {
    social_media_usage: number[];
    academic_performance: number[];
    user_ids: number[];
  };
  correlation: {
    coefficient: number;
    strength: "Débil" | "Moderada" | "Fuerte";
    direction: "Positiva" | "Negativa";
  };
  insights: string[];
  image_path?: string;
}
```

### 4. **GET** `/api/visualizations/platforms` - Distribución de plataformas

**Response:**

```typescript
{
  type: "pie_chart";
  title: string;
  data: {
    platforms: string[];
    user_counts: number[];
    percentages: number[];
  };
  most_popular: {
    platform: string;
    percentage: number;
    user_count: number;
  };
  insights: string[];
  image_path?: string;
}
```

### 5. **GET** `/api/visualizations/performance-by-gender` - Rendimiento por género

**Response:**

```typescript
{
  type: "box_plot";
  title: string;
  data: {
    genders: string[];
    performances: {
      [gender: string]: number[];
    };
    statistics: {
      [gender: string]: {
        mean: number;
        median: number;
        q1: number;
        q3: number;
        min: number;
        max: number;
      };
    };
  };
  insights: string[];
  image_path?: string;
}
```

### 6. **GET** `/api/visualizations/correlations` - Mapa de calor de correlaciones

**Response:**

```typescript
{
  type: "heatmap";
  title: string;
  data: {
    variables: string[];
    correlation_matrix: number[][];
    significant_correlations: Array<{
      var1: string;
      var2: string;
      correlation: number;
      significance: "Alta" | "Media" | "Baja";
    }>;
  };
  insights: string[];
  image_path?: string;
}
```

### 7. **GET** `/api/visualizations/study-hours` - Análisis de horas de estudio

**Response:**

```typescript
{
  type: "bar_chart";
  title: string;
  data: {
    education_levels: string[];
    average_study_hours: number[];
    performance_correlation: number[];
  };
  insights: string[];
  image_path?: string;
}
```

---

## 🚨 MANEJO DE ERRORES

### Códigos de Estado Comunes:

- **200**: Éxito
- **201**: Creado exitosamente
- **400**: Solicitud inválida (datos faltantes/incorrectos)
- **404**: Recurso no encontrado
- **403**: Prohibido (modelo bloqueado)
- **500**: Error interno del servidor

### Estructura de Error:

```typescript
{
  error: string; // Descripción del error
  message?: string; // Mensaje adicional
  details?: object; // Detalles específicos
}
```

---

## 🔧 CONFIGURACIÓN DE FETCH PARA REACT

### Configuración base:

```javascript
const API_BASE = "http://localhost:5000";

const apiCall = async (endpoint, method = "GET", data = null) => {
  const config = {
    method,
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
  };

  if (data) {
    config.body = JSON.stringify(data);
  }

  try {
    const response = await fetch(`${API_BASE}${endpoint}`, config);
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || "Error en la API");
    }

    return result;
  } catch (error) {
    console.error("Error en API:", error);
    throw error;
  }
};
```

### Ejemplos de uso:

```javascript
// Obtener usuarios
const users = await apiCall("/api/users");

// Crear usuario
const newUser = await apiCall("/api/users", "POST", {
  name: "Juan Pérez",
  email: "juan@example.com",
  age: 20,
  gender: "Masculino",
  education_level: "Universidad",
});

// Análisis completo
const analysis = await apiCall(`/api/analyze/user/${userId}`, "POST");

// Dashboard
const dashboard = await apiCall("/api/visualizations/dashboard");
```

---

## 📱 TIPOS TYPESCRIPT RECOMENDADOS

```typescript
// Usuario
interface User {
  id: number;
  name: string;
  email: string;
  age: number;
  gender: "Masculino" | "Femenino" | "Otro";
  education_level: "Bachillerato" | "Universidad" | "Posgrado";
  social_media_usage: number;
  academic_performance: number;
  main_platform: string;
  study_hours: number;
  created_at: string;
}

// Modelo ML
interface MLModel {
  id: number;
  name: string;
  algorithm: string;
  description: string;
  accuracy: number;
  use_cases_list: string[];
  is_locked: boolean;
  unlock_condition: string;
}

// Resultado de análisis
interface AnalysisResult {
  status: "success" | "error";
  timestamp: string;
  results: {
    prediction?: number;
    cluster?: number;
    risk_level?: string;
    interpretation: {
      summary: string;
      key_factors: string[];
      recommendations: string[];
      confidence_level: string;
    };
  };
}

// Visualización
interface Visualization {
  type: string;
  title: string;
  data: object;
  insights: string[];
  image_path?: string;
}
```
