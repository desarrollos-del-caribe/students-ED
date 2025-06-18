from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Crear directorio static si no existe
if not os.path.exists('static'):
    os.makedirs('static')

def load_data():
    try:
        df = pd.read_excel('data/Students_Addiction.xlsx')
        return df
    except Exception as e:
        return {"error": f"Error al cargar el archivo: {str(e)}"}

def clean_data(df):
    df = df.drop_duplicates()
    if df.isnull().sum().any():
        df = df.fillna({
            'Age': df['Age'].mean(),
            'Avg_Daily_Usage_Hours': df['Avg_Daily_Usage_Hours'].mean(),
            'Sleep_Hours_Per_Night': df['Sleep_Hours_Per_Night'].mean(),
            'Mental_Health_Score': df['Mental_Health_Score'].mean(),
            'Conflicts_Over_Social_Media': df['Conflicts_Over_Social_Media'].mean(),
            'Addicted_Score': df['Addicted_Score'].mean(),
            'Gender': 'Unknown',
            'Country': 'Unknown',
            'Most_Used_Platform': 'Unknown',
            'Relationship_Status': 'Unknown'
        })
    return df

def normalize_data(df):
    scaler = MinMaxScaler()
    numeric_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
                    'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Addicted_Score']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

@app.route('/api/validate-age', methods=['GET'])
def validate_age():
    df = load_data()
    if isinstance(df, dict):
        return jsonify(df), 500
    df = clean_data(df)
    info = {
        "min_age": df['Age'].min(),
        "max_age": df['Age'].max(),
        "invalid_ages": df[~((df['Age'] >= 16) & (df['Age'] <= 25))].shape[0]
    }
    return jsonify(info)

@app.route('/api/validate-countries', methods=['GET'])
def validate_countries():
    df = load_data()
    if isinstance(df, dict):
        return jsonify(df), 500
    df = clean_data(df)
    valid_countries = df['Country'].nunique()
    country_counts = df['Country'].value_counts().head(10).to_dict()
    return jsonify({"valid_countries": valid_countries, "country_counts": country_counts})

@app.route('/api/validate-trimester', methods=['GET'])
def validate_trimester():
    df = load_data()
    if isinstance(df, dict):
        return jsonify(df), 500
    df = clean_data(df)
    outliers = df[df['Avg_Daily_Usage_Hours'] > 10].shape[0]
    return jsonify({"outliers": outliers})

@app.route('/api/statistics', methods=['GET'])
def statistics():
    df = load_data()
    if isinstance(df, dict):
        return jsonify(df), 500
    df = clean_data(df)
    stats = df[['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 
                'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Addicted_Score']].describe().to_dict()
    return jsonify(stats)

@app.route('/api/null-info', methods=['GET'])
def null_info():
    df = load_data()
    if isinstance(df, dict):
        return jsonify(df), 500
    df = clean_data(df)
    null_counts = df.isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())
    return jsonify({"null_counts": null_counts, "total_nulls": total_nulls})

@app.route('/api/predict-conflicts', methods=['GET'])
def predict_conflicts():
    df = load_data()
    if isinstance(df, dict):
        return jsonify(df), 500
    df = clean_data(df)
    le_gender = LabelEncoder()
    le_platform = LabelEncoder()
    le_relationship = LabelEncoder()
    df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])
    df['Platform_Encoded'] = le_platform.fit_transform(df['Most_Used_Platform'])
    df['Relationship_Encoded'] = le_relationship.fit_transform(df['Relationship_Status'])
    
    X = df[['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
            'Mental_Health_Score', 'Gender_Encoded', 'Platform_Encoded',
            'Relationship_Encoded', 'Affects_Academic_Performance']]
    y = df['Conflicts_Over_Social_Media'].apply(lambda x: 1 if x >= 3 else 0)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    df['Conflict_Prediction'] = model.predict(X)
    summary = df.groupby('Relationship_Status')['Conflict_Prediction'].mean().to_dict()
    summary_rounded = {k: round(v, 2) for k, v in summary.items()}
    return jsonify(summary_rounded)

@app.route('/api/decision-tree', methods=['GET'])
def decision_tree():
    df = load_data()
    if isinstance(df, dict):
        return jsonify(df), 500
    df = clean_data(df)
    le_gender = LabelEncoder()
    le_platform = LabelEncoder()
    le_relationship = LabelEncoder()
    df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])
    df['Platform_Encoded'] = le_platform.fit_transform(df['Most_Used_Platform'])
    df['Relationship_Encoded'] = le_relationship.fit_transform(df['Relationship_Status'])
    
    X = df[['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
            'Mental_Health_Score', 'Gender_Encoded', 'Platform_Encoded',
            'Relationship_Encoded', 'Affects_Academic_Performance']]
    y = df['Conflicts_Over_Social_Media'].apply(lambda x: 1 if x >= 3 else 0)
    
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X, y)
    
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=X.columns, class_names=['Low Risk', 'High Risk'], filled=True)
    plt.savefig('static/decision_tree.png')
    plt.close()
    
    explanation = """
    El árbol de decisión clasifica a los estudiantes en riesgo alto (Conflicts_Over_Social_Media ≥ 3) o bajo riesgo basado en las siguientes características:
    - Edad
    - Horas de uso diario
    - Horas de sueño
    - Puntuación de salud mental
    - Género, plataforma, estado de relación
    - Impacto académico
    """
    return jsonify({"explanation": explanation, "image": "/static/decision_tree.png"})

@app.route('/api/clustering', methods=['GET'])
def clustering():
    df = load_data()
    if isinstance(df, dict):
        return jsonify(df), 500
    df = clean_data(df)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score']])
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['Avg_Daily_Usage_Hours'], df['Sleep_Hours_Per_Night'], 
                        df['Mental_Health_Score'], c=df['Cluster'], cmap='viridis')
    ax.set_xlabel('Horas de Uso Diario')
    ax.set_ylabel('Horas de Sueño')
    ax.set_zlabel('Puntuación de Salud Mental')
    plt.title('Clustering de Estudiantes')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig('static/clustering_3d.png')
    plt.close()
    
    counts = df['Cluster'].value_counts().to_dict()
    return jsonify({"cluster_counts": counts, "image": "/static/clustering_3d.png"})

@app.route('/api/generate-plots', methods=['GET'])
def generate_plots_endpoint():
    df = load_data()
    if isinstance(df, dict):
        return jsonify(df), 500
    df = clean_data(df)
    
    # Histograma
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Age'], bins=10, kde=True, color='skyblue')
    plt.title('Distribución de Edades')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.savefig('static/age_histogram.png')
    plt.close()
    
    # Gráfico de barras por país
    plt.figure(figsize=(12, 6))
    country_counts = df['Country'].value_counts().head(10)
    sns.barplot(x=country_counts.index, y=country_counts.values, palette='viridis')
    plt.title('Número de Estudiantes por País (Top 10)')
    plt.xlabel('País')
    plt.ylabel('Número de Estudiantes')
    plt.xticks(rotation=45)
    plt.savefig('static/conflicts_by_country.png')
    plt.close()
    
    # Gráfico 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['Avg_Daily_Usage_Hours'], df['Sleep_Hours_Per_Night'], 
              df['Mental_Health_Score'], c=df['Conflicts_Over_Social_Media'], 
              cmap='coolwarm', s=50)
    ax.set_xlabel('Horas de Uso Diario')
    ax.set_ylabel('Horas de Sueño')
    ax.set_zlabel('Puntuación de Salud Mental')
    plt.title('Relación entre Uso, Sueño y Salud Mental')
    plt.savefig('static/3d_plot.png')
    plt.close()
    
    # Gráfico lineal
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Age', y='Avg_Daily_Usage_Hours', hue='Gender', data=df)
    plt.title('Uso Diario de Redes Sociales por Edad y Género')
    plt.xlabel('Edad')
    plt.ylabel('Horas de Uso Diario')
    plt.savefig('static/line_plot.png')
    plt.close()
    
    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Relationship_Status', y='Mental_Health_Score', data=df)
    plt.title('Puntuación de Salud Mental por Estado de Relación')
    plt.xlabel('Estado de Relación')
    plt.ylabel('Puntuación de Salud Mental')
    plt.savefig('static/boxplot.png')
    plt.close()
    
    # Mapa de calor
    plt.figure(figsize=(10, 8))
    numeric_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
                    'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Addicted_Score']
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matriz de Correlación')
    plt.savefig('static/heatmap.png')
    plt.close()
    
    images = {
        "histogram": "/static/age_histogram.png",
        "country_plot": "/static/conflicts_by_country.png",
        "3d_plot": "/static/3d_plot.png",
        "line_plot": "/static/line_plot.png",
        "boxplot": "/static/boxplot.png",
        "heatmap": "/static/heatmap.png"
    }
    
    return jsonify(images)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)