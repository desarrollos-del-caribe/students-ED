# Pruebas desde la terminal
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from excel_service import load_dataset
from analysis_model import predict_social_media_addiction_risk

if __name__ == "__main__":
    result = predict_social_media_addiction_risk(
        usage_hours=5,
        addicted_score=7,
        mental_health_score=4,
        conflicts_score=2
    )
    print("Resultado de la predicción:")
    print(result)
