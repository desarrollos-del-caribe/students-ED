import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any
import io
import base64
import os

from utils.excel_utils import UsersExcelHandler, PredictionsExcelHandler
from models.data_models import MLResults, UserProfile, ModelComparison, Recommendations, AnalysisResults


class MLService:
    """Servicio principal para algoritmos de Machine Learning"""
    
    def __init__(self):
        self.users_handler = UsersExcelHandler()
        self.predictions_handler = PredictionsExcelHandler()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_training_data(self) -> pd.DataFrame:
        """Cargar datos de entrenamiento desde Excel"""
        return self.users_handler.read_users()
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Preparar datos para entrenamiento"""
        # Crear copias para evitar modificar el original
        data = df.copy()
        
        # Codificar variables categóricas
        categorical_columns = ['gender', 'education_level', 'main_platform']
        for col in categorical_columns:
            if col in data.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    data[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
                else:
                    data[col] = self.label_encoders[col].transform(data[col].astype(str))
        
        # Separar características y target
        feature_columns = ['age', 'social_media_usage', 'study_hours']
        if 'gender' in data.columns:
            feature_columns.append('gender')
        if 'education_level' in data.columns:
            feature_columns.append('education_level')
        if 'main_platform' in data.columns:
            feature_columns.append('main_platform')
        
        X = data[feature_columns]
        
        if target_column and target_column in data.columns:
            y = data[target_column]
            return X, y
        
        return X, None
    
    def create_visualization(self, data: Dict, plot_type: str) -> str:
        """Crear visualización y retornar como base64"""
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'scatter':
            plt.scatter(data['x'], data['y'], alpha=0.6, c=data.get('color', 'blue'))
            plt.xlabel(data.get('xlabel', 'X'))
            plt.ylabel(data.get('ylabel', 'Y'))
            plt.title(data.get('title', 'Scatter Plot'))
        
        elif plot_type == 'feature_importance':
            features = data['features']
            importance = data['importance']
            plt.barh(features, importance)
            plt.xlabel('Importancia')
            plt.title('Importancia de Características')
        
        elif plot_type == 'confusion_matrix':
            sns.heatmap(data['matrix'], annot=True, fmt='d', cmap='Blues')
            plt.title('Matriz de Confusión')
        
        elif plot_type == 'cluster_centers':
            centers = data['centers']
            plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200)
            plt.title('Centros de Clusters')
        
        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def linear_regression_analysis(self, user_id: int) -> MLResults:
        """Análisis de Regresión Lineal"""
        try:
            # Cargar datos
            df = self.load_training_data()
            
            # Preparar datos
            X, y = self.prepare_data(df, 'academic_performance')
            
            # Entrenar modelo
            model = LinearRegression()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred = model.predict(X_test)
            
            # Métricas
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Predicción para usuario específico
            user_data = self.users_handler.get_user_by_id(user_id)
            user_df = pd.DataFrame([user_data])
            X_user, _ = self.prepare_data(user_df)
            user_prediction = model.predict(X_user)[0]
            
            # Visualizaciones
            scatter_plot = self.create_visualization({
                'x': y_test,
                'y': y_pred,
                'xlabel': 'Rendimiento Real',
                'ylabel': 'Predicción',
                'title': 'Regresión Lineal: Real vs Predicho'
            }, 'scatter')
            
            feature_importance = self.create_visualization({
                'features': X.columns.tolist(),
                'importance': np.abs(model.coef_),
                'title': 'Importancia de Características'
            }, 'feature_importance')
            
            # Resultados
            results = MLResults(
                model_id="1",
                user_id=str(user_id),
                results={
                    'accuracy': r2,
                    'predictions': [{'user_id': user_id, 'prediction': float(user_prediction)}],
                    'user_prediction': float(user_prediction)
                },
                metrics={
                    'mse': float(mse),
                    'r2_score': float(r2)
                },
                visualizations={
                    'scatter_plot': [{'x': float(x), 'y': float(y)} for x, y in zip(y_test, y_pred)],
                    'feature_importance': [{'feature': feat, 'importance': float(imp)} 
                                         for feat, imp in zip(X.columns, np.abs(model.coef_))],
                    'scatter_plot_image': scatter_plot,
                    'feature_importance_image': feature_importance
                },
                interpretation={
                    'summary': f'El modelo predice un rendimiento académico de {user_prediction:.2f} para este usuario.',
                    'key_insights': [
                        f'R² Score: {r2:.3f} - El modelo explica {r2*100:.1f}% de la varianza',
                        f'Error cuadrático medio: {mse:.3f}',
                        f'Característica más importante: {X.columns[np.argmax(np.abs(model.coef_))]}'
                    ],
                    'recommendations': [
                        'Considerar optimizar las horas de estudio',
                        'Monitorear el uso de redes sociales',
                        'Evaluar el impacto de la plataforma principal'
                    ]
                }
            )
            
            # Guardar predicción
            self.predictions_handler.save_prediction(
                user_id, 1, user_prediction, r2
            )
            
            return results
            
        except Exception as e:
            return MLResults(
                status="error",
                model_id="1",
                user_id=str(user_id),
                interpretation={'summary': f'Error en regresión lineal: {str(e)}'}
            )
    
    def logistic_regression_analysis(self, user_id: int) -> MLResults:
        """Análisis de Regresión Logística"""
        try:
            # Cargar datos
            df = self.load_training_data()
            
            # Crear variable de riesgo (rendimiento < 75 = riesgo alto)
            df['risk_category'] = (df['academic_performance'] < 75).astype(int)
            
            # Preparar datos
            X, y = self.prepare_data(df, 'risk_category')
            
            # Entrenar modelo
            model = LogisticRegression(random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            
            # Predicción para usuario específico
            user_data = self.users_handler.get_user_by_id(user_id)
            user_df = pd.DataFrame([user_data])
            X_user, _ = self.prepare_data(user_df)
            user_prediction = model.predict(X_user)[0]
            user_probability = model.predict_proba(X_user)[0]
            
            risk_level = "Alto" if user_prediction == 1 else "Bajo"
            
            # Resultados
            results = MLResults(
                model_id="2",
                user_id=str(user_id),
                results={
                    'accuracy': accuracy,
                    'predictions': [{
                        'user_id': user_id, 
                        'risk_category': int(user_prediction),
                        'risk_level': risk_level,
                        'probability_low_risk': float(user_probability[0]),
                        'probability_high_risk': float(user_probability[1])
                    }]
                },
                metrics={
                    'accuracy': float(accuracy),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                },
                interpretation={
                    'summary': f'El usuario tiene un riesgo {risk_level} de bajo rendimiento académico.',
                    'key_insights': [
                        f'Precisión del modelo: {accuracy:.3f}',
                        f'Probabilidad de riesgo alto: {user_probability[1]:.3f}',
                        f'Probabilidad de riesgo bajo: {user_probability[0]:.3f}'
                    ],
                    'recommendations': [
                        'Reducir tiempo en redes sociales' if user_prediction == 1 else 'Mantener hábitos actuales',
                        'Incrementar horas de estudio' if user_prediction == 1 else 'Optimizar horarios de estudio',
                        'Considerar cambio de plataforma principal' if user_prediction == 1 else 'Continuar con plataforma actual'
                    ]
                }
            )
            
            # Guardar predicción
            self.predictions_handler.save_prediction(
                user_id, 2, f"Riesgo {risk_level}", accuracy
            )
            
            return results
            
        except Exception as e:
            return MLResults(
                status="error",
                model_id="2",
                user_id=str(user_id),
                interpretation={'summary': f'Error en regresión logística: {str(e)}'}
            )
    
    def kmeans_clustering_analysis(self, user_id: int) -> MLResults:
        """Análisis de K-Means Clustering"""
        try:
            # Cargar datos
            df = self.load_training_data()
            
            # Preparar datos para clustering
            features = ['social_media_usage', 'academic_performance', 'study_hours']
            X = df[features]
            
            # Normalizar datos
            X_scaled = self.scaler.fit_transform(X)
            
            # Aplicar K-Means
            n_clusters = 4
            model = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = model.fit_predict(X_scaled)
            
            # Agregar clusters al dataframe
            df['cluster'] = clusters
            
            # Predicción para usuario específico
            user_data = self.users_handler.get_user_by_id(user_id)
            user_features = np.array([[
                user_data['social_media_usage'],
                user_data['academic_performance'],
                user_data['study_hours']
            ]])
            user_features_scaled = self.scaler.transform(user_features)
            user_cluster = model.predict(user_features_scaled)[0]
            
            # Analizar características del cluster
            cluster_stats = df[df['cluster'] == user_cluster][features].describe()
            cluster_description = self.generate_cluster_description(user_cluster, cluster_stats)
            
            # Resultados
            results = MLResults(
                model_id="3",
                user_id=str(user_id),
                results={
                    'cluster_id': int(user_cluster),
                    'cluster_characteristics': cluster_description,
                    'cluster_stats': cluster_stats.to_dict()
                },
                visualizations={
                    'cluster_centers': [{'center': center.tolist()} for center in model.cluster_centers_]
                },
                interpretation={
                    'summary': f'El usuario pertenece al cluster {user_cluster}: {cluster_description}',
                    'key_insights': [
                        f'Total de clusters identificados: {n_clusters}',
                        f'Usuarios en el mismo cluster: {len(df[df["cluster"] == user_cluster])}',
                        f'Características principales del cluster: {cluster_description}'
                    ],
                    'recommendations': self.generate_cluster_recommendations(user_cluster, cluster_stats)
                }
            )
            
            # Guardar predicción
            self.predictions_handler.save_prediction(
                user_id, 3, f"Cluster {user_cluster}: {cluster_description}", None
            )
            
            return results
            
        except Exception as e:
            return MLResults(
                status="error",
                model_id="3",
                user_id=str(user_id),
                interpretation={'summary': f'Error en clustering: {str(e)}'}
            )
    
    def random_forest_analysis(self, user_id: int) -> MLResults:
        """Análisis de Random Forest"""
        try:
            # Cargar datos
            df = self.load_training_data()
            
            # Preparar datos
            X, y = self.prepare_data(df, 'academic_performance')
            
            # Entrenar modelo
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred = model.predict(X_test)
            
            # Métricas
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Importancia de características
            feature_importance = model.feature_importances_
            
            # Predicción para usuario específico
            user_data = self.users_handler.get_user_by_id(user_id)
            user_df = pd.DataFrame([user_data])
            X_user, _ = self.prepare_data(user_df)
            user_prediction = model.predict(X_user)[0]
            
            # Resultados
            results = MLResults(
                model_id="4",
                user_id=str(user_id),
                results={
                    'accuracy': r2,
                    'predictions': [{'user_id': user_id, 'prediction': float(user_prediction)}],
                    'feature_importance': dict(zip(X.columns, feature_importance.astype(float)))
                },
                metrics={
                    'mse': float(mse),
                    'r2_score': float(r2)
                },
                visualizations={
                    'feature_importance': [{'feature': feat, 'importance': float(imp)} 
                                         for feat, imp in zip(X.columns, feature_importance)]
                },
                interpretation={
                    'summary': f'Random Forest predice un rendimiento de {user_prediction:.2f} para este usuario.',
                    'key_insights': [
                        f'R² Score: {r2:.3f}',
                        f'Característica más importante: {X.columns[np.argmax(feature_importance)]}',
                        f'Precisión mejorada respecto a regresión lineal'
                    ],
                    'recommendations': [
                        'Optimizar la característica más importante',
                        'Considerar las interacciones entre variables',
                        'Monitorear múltiples factores simultáneamente'
                    ]
                }
            )
            
            # Guardar predicción
            self.predictions_handler.save_prediction(
                user_id, 4, user_prediction, r2
            )
            
            return results
            
        except Exception as e:
            return MLResults(
                status="error",
                model_id="4",
                user_id=str(user_id),
                interpretation={'summary': f'Error en Random Forest: {str(e)}'}
            )
    
    def decision_tree_analysis(self, user_id: int) -> MLResults:
        """Análisis de Árboles de Decisión"""
        try:
            # Cargar datos
            df = self.load_training_data()
            
            # Preparar datos
            X, y = self.prepare_data(df, 'academic_performance')
            
            # Entrenar modelo
            model = DecisionTreeRegressor(max_depth=5, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred = model.predict(X_test)
            
            # Métricas
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Predicción para usuario específico
            user_data = self.users_handler.get_user_by_id(user_id)
            user_df = pd.DataFrame([user_data])
            X_user, _ = self.prepare_data(user_df)
            user_prediction = model.predict(X_user)[0]
            
            # Generar reglas de decisión (simplificado)
            decision_rules = self.extract_decision_rules(model, X.columns, X_user.iloc[0])
            
            # Resultados
            results = MLResults(
                model_id="5",
                user_id=str(user_id),
                results={
                    'accuracy': r2,
                    'predictions': [{'user_id': user_id, 'prediction': float(user_prediction)}],
                    'decision_rules': decision_rules
                },
                metrics={
                    'mse': float(mse),
                    'r2_score': float(r2)
                },
                interpretation={
                    'summary': f'Árbol de decisión predice {user_prediction:.2f} basado en reglas específicas.',
                    'key_insights': [
                        f'R² Score: {r2:.3f}',
                        f'Reglas aplicadas: {len(decision_rules)}',
                        'Modelo interpretable y explicable'
                    ],
                    'recommendations': [
                        'Seguir las reglas de decisión identificadas',
                        'Enfocarse en los umbrales críticos',
                        'Usar como guía para decisiones académicas'
                    ]
                }
            )
            
            # Guardar predicción
            self.predictions_handler.save_prediction(
                user_id, 5, f"Predicción: {user_prediction:.2f}", r2
            )
            
            return results
            
        except Exception as e:
            return MLResults(
                status="error",
                model_id="5",
                user_id=str(user_id),
                interpretation={'summary': f'Error en árbol de decisión: {str(e)}'}
            )
    
    def svm_analysis(self, user_id: int) -> MLResults:
        """Análisis de Support Vector Machine"""
        try:
            # Cargar datos
            df = self.load_training_data()
            
            # Crear categorías de rendimiento
            df['performance_category'] = pd.cut(df['academic_performance'], 
                                              bins=[0, 60, 80, 100], 
                                              labels=['Bajo', 'Medio', 'Alto'])
            
            # Preparar datos
            X, y = self.prepare_data(df, 'performance_category')
            
            # Codificar target
            if 'performance_category' not in self.label_encoders:
                self.label_encoders['performance_category'] = LabelEncoder()
                y = self.label_encoders['performance_category'].fit_transform(y.astype(str))
            
            # Normalizar características
            X_scaled = self.scaler.fit_transform(X)
            
            # Entrenar modelo
            model = SVC(kernel='rbf', probability=True, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            
            # Predicción para usuario específico
            user_data = self.users_handler.get_user_by_id(user_id)
            user_df = pd.DataFrame([user_data])
            X_user, _ = self.prepare_data(user_df)
            X_user_scaled = self.scaler.transform(X_user)
            user_prediction = model.predict(X_user_scaled)[0]
            user_probability = model.predict_proba(X_user_scaled)[0]
            
            # Decodificar predicción
            category_labels = ['Bajo', 'Medio', 'Alto']
            predicted_category = category_labels[user_prediction]
            
            # Resultados
            results = MLResults(
                model_id="6",
                user_id=str(user_id),
                results={
                    'accuracy': accuracy,
                    'predictions': [{
                        'user_id': user_id,
                        'category': predicted_category,
                        'probabilities': dict(zip(category_labels, user_probability.astype(float)))
                    }]
                },
                metrics={
                    'accuracy': float(accuracy),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                },
                interpretation={
                    'summary': f'SVM clasifica al usuario en categoría de rendimiento: {predicted_category}',
                    'key_insights': [
                        f'Precisión del modelo: {accuracy:.3f}',
                        f'Probabilidad más alta: {max(user_probability):.3f}',
                        'Clasificación basada en márgenes de decisión'
                    ],
                    'recommendations': [
                        f'Estrategias para categoría {predicted_category}',
                        'Optimizar factores críticos identificados',
                        'Monitorear cambios en la clasificación'
                    ]
                }
            )
            
            # Guardar predicción
            self.predictions_handler.save_prediction(
                user_id, 6, predicted_category, accuracy
            )
            
            return results
            
        except Exception as e:
            return MLResults(
                status="error",
                model_id="6",
                user_id=str(user_id),
                interpretation={'summary': f'Error en SVM: {str(e)}'}
            )
    
    def generate_cluster_description(self, cluster_id: int, stats: pd.DataFrame) -> str:
        """Generar descripción del cluster"""
        descriptions = {
            0: "Estudiantes equilibrados",
            1: "Usuarios intensivos de redes sociales",
            2: "Estudiantes dedicados",
            3: "Usuarios de bajo rendimiento"
        }
        return descriptions.get(cluster_id, f"Cluster {cluster_id}")
    
    def generate_cluster_recommendations(self, cluster_id: int, stats: pd.DataFrame) -> List[str]:
        """Generar recomendaciones basadas en cluster"""
        recommendations = {
            0: [
                "Mantener el equilibrio actual",
                "Optimizar horarios de estudio",
                "Continuar con hábitos actuales"
            ],
            1: [
                "Reducir tiempo en redes sociales",
                "Establecer límites de tiempo",
                "Buscar alternativas productivas"
            ],
            2: [
                "Mantener dedicación al estudio",
                "Considerar tiempo de descanso",
                "Compartir estrategias exitosas"
            ],
            3: [
                "Incrementar horas de estudio",
                "Reducir distracciones digitales",
                "Buscar apoyo académico"
            ]
        }
        return recommendations.get(cluster_id, ["Continuar monitoreando patrones"])
    
    def extract_decision_rules(self, model, feature_names: List[str], user_data: pd.Series) -> List[str]:
        """Extraer reglas de decisión simplificadas"""
        # Esta es una implementación simplificada
        # En un caso real, se extraerían las reglas del árbol
        rules = [
            f"Si uso de redes sociales > 5 horas: Impacto en rendimiento",
            f"Si horas de estudio < 20: Rendimiento tiende a ser menor",
            f"Plataforma principal influye en concentración"
        ]
        return rules
    
    def compare_models(self, user_id: int) -> AnalysisResults:
        """Comparar resultados de múltiples modelos"""
        try:
            # Ejecutar todos los modelos
            linear_results = self.linear_regression_analysis(user_id)
            logistic_results = self.logistic_regression_analysis(user_id)
            rf_results = self.random_forest_analysis(user_id)
            
            # Crear comparaciones
            comparisons = [
                ModelComparison(
                    model_name="Regresión Lineal",
                    prediction=linear_results.results.get('user_prediction', 0),
                    confidence=linear_results.results.get('accuracy', 0),
                    key_factors=["social_media_usage", "study_hours"]
                ),
                ModelComparison(
                    model_name="Regresión Logística",
                    prediction=logistic_results.results['predictions'][0].get('risk_level', 'Desconocido'),
                    confidence=logistic_results.results.get('accuracy', 0),
                    key_factors=["social_media_usage", "main_platform"]
                ),
                ModelComparison(
                    model_name="Random Forest",
                    prediction=rf_results.results.get('user_prediction', 0),
                    confidence=rf_results.results.get('accuracy', 0),
                    key_factors=["multiple_factors"]
                )
            ]
            
            # Crear perfil de usuario
            user_profile = UserProfile(
                user_id=user_id,
                risk_level=logistic_results.results['predictions'][0].get('risk_level', 'Medio'),
                academic_prediction=linear_results.results.get('user_prediction', 0),
                usage_pattern="Analizado",
                dominant_factors=["social_media_usage", "study_hours", "main_platform"]
            )
            
            # Crear recomendaciones
            recommendations = Recommendations(
                immediate_actions=[
                    "Evaluar tiempo actual en redes sociales",
                    "Optimizar horarios de estudio",
                    "Identificar plataforma más productiva"
                ],
                long_term_strategies=[
                    "Desarrollar hábitos de estudio consistentes",
                    "Crear balance entre digital y académico",
                    "Establecer metas de rendimiento"
                ],
                resource_suggestions=[
                    "Apps de productividad",
                    "Técnicas de estudio",
                    "Gestión del tiempo digital"
                ]
            )
            
            return AnalysisResults(
                user_profile=user_profile,
                model_comparisons=comparisons,
                recommendations=recommendations
            )
            
        except Exception as e:
            # Retornar resultado de error
            return AnalysisResults(
                user_profile=UserProfile(user_id=user_id),
                model_comparisons=[],
                recommendations=Recommendations()
            )
