#Visualizaciones para cada modelo - Gráficas
import matplotlib.pyplot as plt
import seaborn as sns
from utils.helpers import save_plot_image_with_timestamp, clean_old_graphs
import os

GRAPH_DIR = os.path.join(os.path.dirname(__file__), '../static/graphs')

#Para predict_mental_health_score ---
def plot_mental_health_comparison(student_score, avg_score):
    plt.figure(figsize=(6, 4))
    plt.bar(['Estudiante', 'Promedio'], [student_score, avg_score], color=['blue', 'gray'])
    plt.title('Comparativa Salud Mental')
    plt.ylabel('Puntaje')
    clean_old_graphs(GRAPH_DIR)
    return save_plot_image_with_timestamp("mental_health")

#Para social_media_addiction_risk ---
def plot_addiction_risk_bar(prob_no_addiction, prob_addiction):
    plt.figure(figsize=(6, 4))
    plt.bar(['No Adicción', 'Adicción'], [prob_no_addiction, prob_addiction], color=['green', 'red'])
    plt.title('Probabilidad de Adicción a Redes')
    plt.ylabel('Probabilidad')
    clean_old_graphs(GRAPH_DIR)
    return save_plot_image_with_timestamp("addiction_risk")

#Para student_performance_prediction ---
def plot_student_performance_comparison(addicted_score, avg_addicted_score, academic_impact, avg_academic_impact):
    plt.figure(figsize=(8, 4))
    labels = ['Addicted Score', 'Academic Impact']
    student = [addicted_score, academic_impact]
    promedio = [avg_addicted_score, avg_academic_impact]

    x = range(len(labels))
    width = 0.35
    plt.bar([i - width/2 for i in x], student, width=width, label='Estudiante', color='blue')
    plt.bar([i + width/2 for i in x], promedio, width=width, label='Promedio', color='gray')
    plt.xticks(x, labels)
    plt.title('Comparación: Estudiante vs Promedio')
    plt.ylabel('Valor')
    plt.legend()
    clean_old_graphs(GRAPH_DIR)
    return save_plot_image_with_timestamp("student_performance")

#Para academic_performance_risk ---
def plot_academic_risk_pie(probability):
    plt.figure(figsize=(5, 5))
    plt.pie([1 - probability, probability], labels=['Bajo', 'Alto'], colors=['green', 'red'], autopct='%1.1f%%')
    plt.title('Riesgo Académico')
    clean_old_graphs(GRAPH_DIR)
    return save_plot_image_with_timestamp("academic_risk")

#Para addiction_by_country ---
def plot_addiction_by_country(countries, avg_scores):
    plt.figure(figsize=(10, 5))
    plt.bar(countries, avg_scores, color='purple')
    plt.title('Adicción por País')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Puntaje Promedio de Adicción')
    clean_old_graphs(GRAPH_DIR)
    return save_plot_image_with_timestamp("addiction_country")

#Para heatmap ---
def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 6))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Mapa de Calor de Correlaciones')
    clean_old_graphs(GRAPH_DIR)
    return save_plot_image_with_timestamp("correlation_heatmap")
