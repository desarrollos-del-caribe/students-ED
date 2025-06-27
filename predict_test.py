import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analysis_models import (
    social_media_addiction_risk, logistic_regression_prediction, 
    kmeans_clustering, correlation_analysis, academic_level_distribution, sleep_comparison_by_profile,
    predict_academic_performance, platform_usage_distribution, prediction_sleep
)

def main():
    print("===== Predicciones disponibles =====")
    print("1. Árbol de decisión: Análisis de riesgo de adicción a redes sociales")
    print("2. Regresión lineal: ¿Cuántas horas de sueño deberías tener?")
    print("3. Regresión logística: ¿Tienes riesgo de conflictos por redes sociales?")
    print("4. K-Means: ¿Qué tipo de usuario eres?")
    print("5. Correlación: ¿Qué factores afectan tu rendimiento?")
    print("6. Usuarios por nivel académico:")
    print("7. ¿Duermes lo suficiente según tu perfil?")
    print("8. Predicción general de rendimiento académico")
    print("9. Redes sociales mas usadas")
    print("0. Salir")
    opcion = input("Selecciona una opción: ")

    match opcion:
        case "1":
            social_media_addiction_risk()
        case "2":
            prediction_sleep()
        case "3":
            logistic_regression_prediction()
        case "4":
            kmeans_clustering()
        case "5":
            correlation_analysis()
        case "6":
            academic_level_distribution()
        case "7":
            sleep_comparison_by_profile()
        case "8":
            predict_academic_performance()
        case "9":
            platform_usage_distribution()
        case "0":
            print("Saliendo...")
        case _:
            print("Opción inválida")


if __name__ == "__main__":
    main()
