import logging
import pandas as pd
import os
import matplotlib.pyplot as plt
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
        # Renombrar columnas para coincidir con el código
        df = df.rename(columns={
            'Avg_Daily_Usage_Hours': 'Social Media Usage',
            'Sleep_Hours_Per_Night': 'Sleep Hours',
            'Mental_Health_Score': 'Stress Level',
            'Addicted_Score': 'Life Satisfaction',
            'Affects_Academic_Performance': 'Academic Performance',
            'Most_Used_Platform': 'Social Media Platform'
        })
        # Convertir columnas a numérico, reemplazando errores con NaN
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['Social Media Usage'] = pd.to_numeric(df['Social Media Usage'], errors='coerce')
        df['Sleep Hours'] = pd.to_numeric(df['Sleep Hours'], errors='coerce')
        df['Stress Level'] = pd.to_numeric(df['Stress Level'], errors='coerce')
        df['Life Satisfaction'] = pd.to_numeric(df['Life Satisfaction'], errors='coerce')
        df['Academic Performance'] = pd.to_numeric(df['Academic Performance'], errors='coerce')
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
        country_counts = df['Country'].value_counts().head(10).to_dict()
        return {"data": {"countries": {"valid_countries": valid_countries, "country_counts": country_counts}}}, 200
    except Exception as e:
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
        if 'Social Media Usage' not in df.columns:
            return jsonify({"error": "Columna 'Social Media Usage' no encontrada en el dataset"}), 400
        Q1 = df['Social Media Usage'].quantile(0.25)
        Q3 = df['Social Media Usage'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = int(((df['Social Media Usage'] < (Q1 - 1.5 * IQR)) | (df['Social Media Usage'] > (Q3 + 1.5 * IQR))).sum())
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
        if 'Age' not in df.columns or 'Social Media Usage' not in df.columns:
            return jsonify({"error": "Columnas 'Age' o 'Social Media Usage' no encontradas en el dataset"}), 400
        if not pd.api.types.is_numeric_dtype(df['Age']) or not pd.api.types.is_numeric_dtype(df['Social Media Usage']):
            return jsonify({"error": "Las columnas 'Age' o 'Social Media Usage' no contienen datos numéricos"}), 400
        
        plots_dir = os.path.join(os.path.dirname(__file__), 'static', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Histograma de edades
        plt.figure(figsize=(10, 6))
        df['Age'].hist(bins=20)
        plt.title('Distribución de Edades')
        plt.xlabel('Edad')
        plt.ylabel('Frecuencia')
        age_hist_path = os.path.join(plots_dir, 'age_histogram.png')
        plt.savefig(age_hist_path)
        plt.close()
        
        # Gráfico de dispersión
        plt.figure(figsize=(10, 6))
        plt.scatter(df['Age'], df['Social Media Usage'])
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
        required_cols = ['Social Media Usage', 'Stress Level', 'Life Satisfaction']
        if not all(col in df.columns for col in required_cols):
            return jsonify({"error": f"Columnas requeridas no encontradas: {', '.join(required_cols)}"}), 400
        X = df[['Social Media Usage', 'Stress Level']]
        y = df['Life Satisfaction']
        model = LinearRegression()
        model.fit(X, y)
        coefficients = {f'coef_{col}': float(val) for col, val in zip(X.columns, model.coef_)}
        intercept = float(model.intercept_)
        r2_score = float(model.score(X, y))
        return {"data": {"linear_regression": {"coefficients": coefficients, "intercept": intercept, "r2_score": r2_score}}}, 200
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
        required_cols = ['Social Media Usage', 'Stress Level', 'Sleep Hours']
        if not all(col in df.columns for col in required_cols):
            return jsonify({"error": f"Columnas requeridas no encontradas: {', '.join(required_cols)}"}), 400
        df['High_Stress'] = (df['Stress Level'] > df['Stress Level'].median()).astype(int)
        X = df[['Social Media Usage', 'Sleep Hours']]
        y = df['High_Stress']
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        coefficients = {f'coef_{col}': float(val) for col, val in zip(X.columns, model.coef_[0])}
        intercept = float(model.intercept_[0])
        accuracy = float(model.score(X, y))
        return {"data": {"logistic_regression": {"coefficients": coefficients, "intercept": intercept, "accuracy": accuracy}}}, 200
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
        return {"data": {"correlation_matrix": correlation_matrix}}, 200
    except Exception as e:
        return jsonify({"error": f"Error en correlation_analysis: {str(e)}"}), 500

def decision_tree_analysis():
    df = load_data()
    if isinstance(df, tuple):
        return df
    try:
        df = clean_data(df)
        if isinstance(df, tuple):
            return df
        required_cols = ['Social Media Usage', 'Age', 'Sleep Hours', 'Stress Level']
        if not all(col in df.columns for col in required_cols):
            return jsonify({"error": f"Columnas requeridas no encontradas: {', '.join(required_cols)}"}), 400
        df['High_Social_Media'] = (df['Social Media Usage'] > df['Social Media Usage'].median()).astype(int)
        X = df[['Age', 'Sleep Hours', 'Stress Level']]
        y = df['High_Social_Media']
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X, y)
        feature_importance = {f'importance_{col}': float(val) for col, val in zip(X.columns, model.feature_importances_)}
        accuracy = float(model.score(X, y))

        # Generar imagen del árbol de decisión
        plots_dir = os.path.join(os.path.dirname(__file__), 'static', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        tree_path = os.path.join(plots_dir, 'decision_tree.png')

        # Verificar permisos de escritura
        if not os.access(plots_dir, os.W_OK):
            logger.error(f"No hay permisos de escritura en {plots_dir}")
            return jsonify({"error": f"No hay permisos de escritura en {plots_dir}"}), 500

        # Generar árbol con matplotlib
        try:
            plt.figure(figsize=(40, 15))  # Tamaño ajustado para legibilidad
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

        # Verificar si el archivo se generó y es accesible
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

def anova_analysis():
    df = load_data()
    if isinstance(df, tuple):
        return df
    try:
        df = clean_data(df)
        if isinstance(df, tuple):
            return df
        required_cols = ['Social Media Platform', 'Stress Level']
        if not all(col in df.columns for col in required_cols):
            return jsonify({"error": f"Columnas requeridas no encontradas: {', '.join(required_cols)}"}), 400
        groups = [df[df['Social Media Platform'] == platform]['Stress Level'] for platform in df['Social Media Platform'].unique()]
        f_statistic, p_value = f_oneway(*groups)
        return {"data": {"anova": {"f_statistic": float(f_statistic), "p_value": float(p_value)}}}, 200
    except Exception as e:
        return jsonify({"error": f"Error en anova_analysis: {str(e)}"}), 500