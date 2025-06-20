import logging
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import f_oneway
from flask import jsonify
from sklearn.tree import plot_tree

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_data():
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'data', 'Students_Addiction.xlsx')
        df = pd.read_excel(file_path)
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['Avg_Daily_Usage_Hours'] = pd.to_numeric(df['Avg_Daily_Usage_Hours'], errors='coerce')
        df['Sleep_Hours_Per_Night'] = pd.to_numeric(df['Sleep_Hours_Per_Night'], errors='coerce')
        df['Mental_Health_Score'] = pd.to_numeric(df['Mental_Health_Score'], errors='coerce')
        df['Addicted_Score'] = pd.to_numeric(df['Addicted_Score'], errors='coerce')
        df['Affects_Academic_Performance'] = pd.to_numeric(df['Affects_Academic_Performance'], errors='coerce')
        df['Conflicts_Over_Social_Media'] = pd.to_numeric(df['Conflicts_Over_Social_Media'], errors='coerce')
        return df
    except FileNotFoundError:
        return jsonify({"error": "Archivo Students_Addiction.xlsx no encontrado en la carpeta data"}), 404
    except Exception as e:
        return jsonify({"error": f"Error al cargar el archivo: {str(e)}"}), 500

def clean_data(df):
    try:
        df = df.dropna()
        if df.empty:
            return jsonify({"error": "No hay datos válidos después de eliminar nulos"}), 400
        return df
    except Exception as e:
        return jsonify({"error": f"Error al limpiar los datos: {str(e)}"}), 500

def validate_age():
    df = load_data()
    if isinstance(df, tuple):
        return df
    try:
        df = clean_data(df)
        if isinstance(df, tuple):
            return df
        if 'Age' not in df.columns:
            return jsonify({"error": "Columna 'Age' no encontrada en el dataset"}), 400
        min_age = int(df['Age'].min())
        max_age = int(df['Age'].max())
        invalid_ages = int(((df['Age'] < 16) | (df['Age'] > 25)).sum())
        return {"data_age": {"age_info": {"min_age": min_age, "max_age": max_age, "invalid_ages": invalid_ages}}}, 200
    except Exception as e:
        return jsonify({"error": f"Error en validate_age: {str(e)}"}), 500

def validate_countries():
    df = load_data()
    if isinstance(df, tuple):
        return df
    try:
        df = clean_data(df)
        if isinstance(df, tuple):
            return df
        if 'Country' not in df.columns:
            return jsonify({"error": "Columna 'Country' no encontrada en el dataset"}), 400

        valid_countries = int(df['Country'].nunique())
        
        # Serie de conteos para graficar
        country_series = df['Country'].value_counts().head(10)

        # Diccionario para el JSON
        country_counts = country_series.to_dict()

        # === Generar gráfica ===
        plots_dir = os.path.join(os.path.dirname(__file__), 'static', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        filename = "countries_graph.png"
        graph_path = os.path.join(plots_dir, filename)

        plt.figure(figsize=(12, 6))
        country_series.plot(kind='bar', color='skyblue')
        plt.title('Top 10 países con más estudiantes')
        plt.xlabel('País')
        plt.ylabel('Cantidad de estudiantes')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(graph_path)
        plt.close()

        return {
            "data": {
                "countries": {
                    "valid_countries": valid_countries,
                    "country_counts": country_counts,
                    "graph": f"/static/plots/{filename}"
                }
            }
        }, 200

    except Exception as e:
        logger.error(f"Error en validate_countries: {str(e)}")
        return jsonify({"error": f"Error en validate_countries: {str(e)}"}), 500

def get_null_info():
    df = load_data()
    if isinstance(df, tuple):
        return df
    try:
        null_counts = df.isnull().sum().to_dict()
        total_nulls = int(df.isnull().sum().sum())
        return {"data": {"count": {"null_counts": null_counts, "total_nulls": total_nulls}}}, 200
    except Exception as e:
        return jsonify({"error": f"Error en get_null_info: {str(e)}"}), 500

def get_statistics():
    df = load_data()
    if isinstance(df, tuple):
        return df
    try:
        df = clean_data(df)
        if isinstance(df, tuple):
            return df
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) == 0:
            return jsonify({"error": "No hay columnas numéricas en el dataset"}), 400
        stats_dict = df[numeric_cols].describe().to_dict()
        return {"data": {"stats_dict": stats_dict}}, 200
    except Exception as e:
        return jsonify({"error": f"Error en get_statistics: {str(e)}"}), 500

def detect_outliers():
    df = load_data()
    if isinstance(df, tuple):
        return df
    try:
        df = clean_data(df)
        if isinstance(df, tuple):
            return df
        if 'Avg_Daily_Usage_Hours' not in df.columns:
            return jsonify({"error": "Columna 'Avg_Daily_Usage_Hours' no encontrada en el dataset"}), 400
        Q1 = df['Avg_Daily_Usage_Hours'].quantile(0.25)
        Q3 = df['Avg_Daily_Usage_Hours'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = int(((df['Avg_Daily_Usage_Hours'] < (Q1 - 1.5 * IQR)) | (df['Avg_Daily_Usage_Hours'] > (Q3 + 1.5 * IQR))).sum())
        return {"data": {"outliers": outliers}}, 200
    except Exception as e:
        return jsonify({"error": f"Error en detect_outliers: {str(e)}"}), 500

def generate_plots():
    df = load_data()
    if isinstance(df, tuple):
        return df
    try:
        df = clean_data(df)
        if isinstance(df, tuple):
            return df
        if 'Age' not in df.columns or 'Avg_Daily_Usage_Hours' not in df.columns:
            return jsonify({"error": "Columnas 'Age' o 'Avg_Daily_Usage_Hours' no encontradas en el dataset"}), 400
        if not pd.api.types.is_numeric_dtype(df['Age']) or not pd.api.types.is_numeric_dtype(df['Avg_Daily_Usage_Hours']):
            return jsonify({"error": "Las columnas 'Age' o 'Avg_Daily_Usage_Hours' no contienen datos numéricos"}), 400
        
        plots_dir = os.path.join(os.path.dirname(__file__), 'static', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        df['Age'].hist(bins=20)
        plt.title('Distribución de Edades')
        plt.xlabel('Edad')
        plt.ylabel('Frecuencia')
        age_hist_path = os.path.join(plots_dir, 'age_histogram.png')
        plt.savefig(age_hist_path)
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(df['Age'], df['Avg_Daily_Usage_Hours'])
        plt.title('Uso de Redes Sociales vs Edad')
        plt.xlabel('Edad')
        plt.ylabel('Uso de Redes Sociales (horas)')
        scatter_path = os.path.join(plots_dir, 'social_media_scatter.png')
        plt.savefig(scatter_path)
        plt.close()
        
        plot_urls = [
            f"/static/plots/age_histogram.png",
            f"/static/plots/social_media_scatter.png"
        ]
        return {"data": {"plots": plot_urls}}, 200
    except PermissionError:
        return jsonify({"error": "Permiso denegado al guardar las imágenes en static/plots"}), 500
    except Exception as e:
        return jsonify({"error": f"Error en generate_plots: {str(e)}"}), 500

def linear_regression_analysis():
    df = load_data()
    if isinstance(df, tuple):
        return df
    try:
        df = clean_data(df)
        if isinstance(df, tuple):
            return df
        required_cols = ['Avg_Daily_Usage_Hours', 'Mental_Health_Score', 'Addicted_Score']
        if not all(col in df.columns for col in required_cols):
            return jsonify({"error": f"Columnas requeridas no encontradas: {', '.join(required_cols)}"}), 400
        
        X = df[['Avg_Daily_Usage_Hours', 'Mental_Health_Score']]
        y = df['Addicted_Score']
        model = LinearRegression()
        model.fit(X, y)
        coefficients = {f'coef_{col}': float(val) for col, val in zip(X.columns, model.coef_)}
        intercept = float(model.intercept_)
        r2_score = float(model.score(X, y))

        plots_dir = os.path.join(os.path.dirname(__file__), 'static', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, 'linear_regression.png')

        if not os.access(plots_dir, os.W_OK):
            logger.error(f"No hay permisos de escritura en {plots_dir}")
            return jsonify({"error": f"No hay permisos de escritura en {plots_dir}"}), 500

        try:
            plt.figure(figsize=(10, 6))
            # Plot scatter of Avg_Daily_Usage_Hours vs Addicted_Score, colored by Mental_Health_Score
            scatter = plt.scatter(
                df['Avg_Daily_Usage_Hours'],
                df['Addicted_Score'],
                c=df['Mental_Health_Score'],
                cmap='viridis',
                alpha=0.7
            )
            plt.colorbar(scatter, label='Puntuación de Salud Mental')
            
            # Calculate regression line using only Avg_Daily_Usage_Hours for simplicity
            x_vals = [min(df['Avg_Daily_Usage_Hours']), max(df['Avg_Daily_Usage_Hours'])]
            # Use mean Mental_Health_Score for the regression line to reduce dimensionality
            mean_mental_health = df['Mental_Health_Score'].mean()
            y_vals = [
                intercept + coefficients['coef_Avg_Daily_Usage_Hours'] * x + 
                coefficients['coef_Mental_Health_Score'] * mean_mental_health
                for x in x_vals
            ]
            plt.plot(x_vals, y_vals, color='red', label='Línea de Regresión')
            
            plt.xlabel('Uso de Redes Sociales (horas)')
            plt.ylabel('Puntuación de Adicción')
            plt.title('Regresión Lineal: Adicción vs. Uso de Redes')
            plt.legend()
            
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Error al generar el gráfico de regresión lineal: {str(e)}")
            return jsonify({"error": f"Error al generar el gráfico de regresión lineal: {str(e)}"}), 500

        if not os.path.exists(plot_path) or os.path.getsize(plot_path) == 0:
            logger.error(f"El archivo {plot_path} no se generó correctamente o está vacío")
            return jsonify({"error": f"El archivo del gráfico de regresión lineal no se generó correctamente"}), 500

        logger.info(f"Gráfico de regresión lineal generado correctamente en {plot_path}")

        return {
            "data": {
                "linear_regression": {
                    "coefficients": coefficients,
                    "intercept": intercept,
                    "r2_score": r2_score,
                    "plot_image": "/static/plots/linear_regression.png"
                }
            }
        }, 200
    except Exception as e:
        return jsonify({"error": f"Error en linear_regression_analysis: {str(e)}"}), 500

def logistic_regression_analysis():
    df = load_data()
    if isinstance(df, tuple):
        return df
    try:
        df = clean_data(df)
        if isinstance(df, tuple):
            return df
        required_cols = ['Avg_Daily_Usage_Hours', 'Mental_Health_Score', 'Sleep_Hours_Per_Night']
        if not all(col in df.columns for col in required_cols):
            return jsonify({"error": f"Columnas requeridas no encontradas: {', '.join(required_cols)}"}), 400
        
        df['High_Stress'] = (df['Mental_Health_Score'] > df['Mental_Health_Score'].median()).astype(int)
        X = df[['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night']]
        y = df['High_Stress']
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        coefficients = {f'coef_{col}': float(val) for col, val in zip(X.columns, model.coef_[0])}
        intercept = float(model.intercept_[0])
        accuracy = float(model.score(X, y))

        plots_dir = os.path.join(os.path.dirname(__file__), 'static', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, 'logistic_regression.png')

        if not os.access(plots_dir, os.W_OK):
            logger.error(f"No hay permisos de escritura en {plots_dir}")
            return jsonify({"error": f"No hay permisos de escritura en {plots_dir}"}), 500

        try:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                x=df['Avg_Daily_Usage_Hours'],
                y=df['Sleep_Hours_Per_Night'],
                hue=df['High_Stress'].map({0: 'Bajo Estrés', 1: 'Alto Estrés'}),
                style=df['High_Stress'],
                palette='deep',
                alpha=0.7
            )

            # Create a grid for decision boundary without numpy
            x_min, x_max = min(df['Avg_Daily_Usage_Hours']), max(df['Avg_Daily_Usage_Hours'])
            y_min, y_max = min(df['Sleep_Hours_Per_Night']), max(df['Sleep_Hours_Per_Night'])
            x_step = (x_max - x_min) / 100
            y_step = (y_max - y_min) / 100
            grid_points = []
            grid_predictions = []
            
            for x in [x_min + i * x_step for i in range(100)]:
                for y in [y_min + j * y_step for j in range(100)]:
                    grid_points.append([x, y])
                    # Logistic function: 1 / (1 + exp(-(b0 + b1*x + b2*y)))
                    z = intercept + coefficients['coef_Avg_Daily_Usage_Hours'] * x + coefficients['coef_Sleep_Hours_Per_Night'] * y
                    prob = 1 / (1 + 2.71828 ** (-z))  # Using math constant e ≈ 2.71828
                    grid_predictions.append(1 if prob >= 0.5 else 0)
            
            # Reshape predictions into a grid-like structure for contour plotting
            grid_x = [x_min + i * x_step for i in range(100)]
            grid_y = [y_min + j * y_step for j in range(100)]
            grid_z = []
            idx = 0
            for _ in range(100):
                row = []
                for _ in range(100):
                    row.append(grid_predictions[idx])
                    idx += 1
                grid_z.append(row)
            
            plt.contourf(grid_x, grid_y, grid_z, alpha=0.3, cmap='coolwarm')

            plt.xlabel('Uso de Redes Sociales (horas)')
            plt.ylabel('Horas de Sueño por Noche')
            plt.title('Regresión Logística: Clasificación de Estrés')
            plt.legend(title='Nivel de Estrés')
            
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Error al generar el gráfico de regresión logística: {str(e)}")
            return jsonify({"error": f"Error al generar el gráfico de regresión logística: {str(e)}"}), 500

        if not os.path.exists(plot_path) or os.path.getsize(plot_path) == 0:
            logger.error(f"El archivo {plot_path} no se generó correctamente o está vacío")
            return jsonify({"error": f"El archivo del gráfico de regresión logística no se generó correctamente"}), 500

        logger.info(f"Gráfico de regresión logística generado correctamente en {plot_path}")

        return {
            "data": {
                "logistic_regression": {
                    "coefficients": coefficients,
                    "intercept": intercept,
                    "accuracy": accuracy,
                    "plot_image": "/static/plots/logistic_regression.png"
                }
            }
        }, 200
    except Exception as e:
        return jsonify({"error": f"Error en logistic_regression_analysis: {str(e)}"}), 500

def correlation_analysis():
    df = load_data()
    if isinstance(df, tuple):
        return df
    try:
        df = clean_data(df)
        if isinstance(df, tuple):
            return df
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) == 0:
            return jsonify({"error": "No hay columnas numéricas para correlación"}), 400
        correlation_matrix = df[numeric_cols].corr().to_dict()
        
        plots_dir = os.path.join(os.path.dirname(__file__), 'static', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        heatmap_path = os.path.join(plots_dir, 'correlation_heatmap.png')

        if not os.access(plots_dir, os.W_OK):
            logger.error(f"No hay permisos de escritura en {plots_dir}")
            return jsonify({"error": f"No hay permisos de escritura en {plots_dir}"}), 500

        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                df[numeric_cols].corr(),
                annot=True,
                cmap='coolwarm',
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                fmt='.2f'
            )
            plt.title('Mapa de Calor de Correlación', fontsize=14, pad=20)
            plt.savefig(heatmap_path, dpi=100, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Error al generar el mapa de calor: {str(e)}")
            return jsonify({"error": f"Error al generar el mapa de calor: {str(e)}"}), 500

        if not os.path.exists(heatmap_path) or os.path.getsize(heatmap_path) == 0:
            logger.error(f"El archivo {heatmap_path} no se generó correctamente o está vacío")
            return jsonify({"error": f"El archivo del mapa de calor no se generó correctamente"}), 500

        logger.info(f"Mapa de calor de correlación generado correctamente en {heatmap_path}")
        return {
            "data": {
                "correlation_matrix": correlation_matrix,
                "heatmap_image": "/static/plots/correlation_heatmap.png"
            }
        }, 200
    except Exception as e:
        logger.error(f"Error en correlation_analysis: {str(e)}")
        return jsonify({"error": f"Error en correlation_analysis: {str(e)}"}), 500

def decision_tree_analysis():
    df = load_data()
    if isinstance(df, tuple):
        return df
    try:
        df = clean_data(df)
        if isinstance(df, tuple):
            return df
        required_cols = ['Avg_Daily_Usage_Hours', 'Age', 'Sleep_Hours_Per_Night', 'Mental_Health_Score']
        if not all(col in df.columns for col in required_cols):
            return jsonify({"error": f"Columnas requeridas no encontradas: {', '.join(required_cols)}"}), 400
        df['High_Social_Media'] = (df['Avg_Daily_Usage_Hours'] > df['Avg_Daily_Usage_Hours'].median()).astype(int)
        X = df[['Age', 'Sleep_Hours_Per_Night', 'Mental_Health_Score']]
        y = df['High_Social_Media']
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X, y)
        feature_importance = {f'importance_{col}': float(val) for col, val in zip(X.columns, model.feature_importances_)}
        accuracy = float(model.score(X, y))

        plots_dir = os.path.join(os.path.dirname(__file__), 'static', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        tree_path = os.path.join(plots_dir, 'decision_tree.png')

        if not os.access(plots_dir, os.W_OK):
            logger.error(f"No hay permisos de escritura en {plots_dir}")
            return jsonify({"error": f"No hay permisos de escritura en {plots_dir}"}), 500

        try:
            plt.figure(figsize=(40, 15))
            plot_tree(
                model,
                feature_names=X.columns,
                class_names=['Low Social Media', 'High Social Media'],
                filled=True,
                rounded=True,
                proportion=True,
                impurity=True,
                fontsize=10
            )
            plt.title('Árbol de Decisión', fontsize=14, pad=20)
            plt.savefig(tree_path, dpi=100, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Error al generar la imagen del árbol: {str(e)}")
            return jsonify({"error": f"Error al generar la imagen del árbol: {str(e)}"}), 500

        if not os.path.exists(tree_path) or os.path.getsize(tree_path) == 0:
            logger.error(f"El archivo {tree_path} no se generó correctamente o está vacío")
            return jsonify({"error": f"El archivo de imagen del árbol no se generó correctamente"}), 500

        logger.info(f"Árbol de decisión generado correctamente en {tree_path}")
        return {
            "data": {
                "decision_tree": {
                    "feature_importance": feature_importance,
                    "accuracy": accuracy,
                    "tree_image": "/static/plots/decision_tree.png"
                }
            }
        }, 200
    except Exception as e:
        logger.error(f"Error en decision_tree_analysis: {str(e)}")
        return jsonify({"error": f"Error en decision_tree_analysis: {str(e)}"}), 500