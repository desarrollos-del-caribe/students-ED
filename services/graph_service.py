#Visualizaciones para cada modelo - Gráficas
import matplotlib.pyplot as plt
import seaborn as sns
from utils.helpers import save_plot_image_with_timestamp, clean_old_graphs
import os

GRAPH_DIR = os.path.join(os.path.dirname(__file__), '../static/graphs')

#Visualizaciones para cada modelo - Gráficas

def plot_mental_health_comparison(student_score, avg_score):
    return {
        "x": ['Estudiante', 'Promedio'],
        "y": [student_score, avg_score],
        "label": "Comparativa Salud Mental"
    }

def plot_addiction_risk_bar(prob_no_addiction, prob_addiction):
    return {
        "x": ['No Adicción', 'Adicción'],
        "y": [prob_no_addiction, prob_addiction],
        "label": "Probabilidad de Adicción a Redes"
    }

def plot_student_performance_comparison(addicted_score, avg_addicted_score, academic_impact, avg_academic_impact):
    return {
        "labels": ['Addicted Score', 'Academic Impact'],
        "student": [addicted_score, academic_impact],
        "promedio": [avg_addicted_score, avg_academic_impact],
        "label": "Comparación Estudiante vs Promedio"
    }

def plot_academic_risk_pie(probability):
    return {
        "labels": ['Bajo', 'Alto'],
        "values": [1 - probability, probability],
        "label": "Riesgo Académico"
    }

def plot_addiction_by_country(countries, avg_scores):
    return {
        "x": countries,
        "y": avg_scores,
        "label": "Adicción por País"
    }

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr().round(2)
    return {
        "matrix": corr.values.tolist(),
        "labels": corr.columns.tolist(),
        "label": "Mapa de Calor de Correlaciones"
    }

