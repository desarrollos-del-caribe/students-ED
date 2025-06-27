from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services import load_data, clean_data  # Asegúrate de que estas funciones estén bien importadas

# === Datos simulados del usuario ===
simulated_user = {
    'Age': 21,
    'Gender': 'Male',
    'Academic_Level': 'Undergraduate',
    'Country': 'Mexico',
    'Avg_Daily_Usage_Hours': 5.8,
    'Most_Used_Platform': 'YouTube',
    'Affects_Academic_Performance': 1,
    'Sleep_Hours_Per_Night': 6.2,
    'Mental_Health_Score': 6,
    'Relationship_Status': 'Single',
    'Conflicts_Over_Social_Media': 2,
    'Addicted_Score': 7
}

def social_media_addiction_risk():
    print("\n📌 Análisis de riesgo de adicción a redes sociales\n")

    # Cargar y limpiar datos
    df = load_data()
    df = clean_data(df)

    # Columnas necesarias
    required_cols = ['Avg_Daily_Usage_Hours', 'Addicted_Score', 'Mental_Health_Score', 'Conflicts_Over_Social_Media']
    if not all(col in df.columns for col in required_cols):
        print("❌ Columnas requeridas no encontradas.")
        return

    # Clasificar niveles de riesgo (etiqueta a predecir)
    def clasificar_riesgo(row):
        if row['Avg_Daily_Usage_Hours'] > 5 and row['Addicted_Score'] >= 7 and row['Conflicts_Over_Social_Media'] >= 2:
            return "Alto"
        elif row['Addicted_Score'] >= 5 or row['Mental_Health_Score'] <= 5:
            return "Medio"
        else:
            return "Bajo"

    df['Riesgo_Adiccion'] = df.apply(clasificar_riesgo, axis=1)

    # Preparar datos para modelo
    X = df[required_cols]
    y = df['Riesgo_Adiccion']

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)

    # Predicción del usuario
    entrada = pd.DataFrame([{
        'Avg_Daily_Usage_Hours': simulated_user['Avg_Daily_Usage_Hours'],
        'Addicted_Score': simulated_user['Addicted_Score'],
        'Mental_Health_Score': simulated_user['Mental_Health_Score'],
        'Conflicts_Over_Social_Media': simulated_user['Conflicts_Over_Social_Media']
    }])

    pred = model.predict(entrada)[0]

    print(f"📊 Según tu perfil, tu riesgo de adicción a redes sociales es: **{pred.upper()}**\n")

    if pred == "Alto":
        print("🔴 Riesgo Alto: Usas redes muchas horas, presentas conflictos y una puntuación alta de adicción.")
        print("❗ Recomendación: Establece límites diarios, usa apps de control de tiempo y busca apoyo si sientes dependencia.\n")
    elif pred == "Medio":
        print("🟠 Riesgo Medio: Tienes hábitos regulares, pero muestras algunas señales preocupantes.")
        print("📌 Consejo: Observa tus rutinas y procura balancear el uso con otras actividades.\n")
    else:
        print("🟢 Riesgo Bajo: No presentas indicadores de adicción. ¡Buen manejo!")
        print("✅ Continúa con hábitos saludables y equilibrio digital.\n")

def main():
    decision_tree_prediction()

if __name__ == "__main__":
    main()


def prediction_sleep():
    print("\n💤 Regresión Lineal: ¿Duermes lo suficiente según tu perfil?\n")

    df = load_data()
    df = clean_data(df)

    features = ['Age', 'Avg_Daily_Usage_Hours', 'Mental_Health_Score', 'Sleep_Hours_Per_Night']
    if not all(col in df.columns for col in features):
        print("❌ Columnas necesarias no encontradas.")
        return

    # Preparar datos
    X = df[['Age', 'Avg_Daily_Usage_Hours', 'Mental_Health_Score']]
    y = df['Sleep_Hours_Per_Night']

    model = LinearRegression()
    model.fit(X, y)

    entrada = pd.DataFrame([{
        'Age': simulated_user['Age'],
        'Avg_Daily_Usage_Hours': simulated_user['Avg_Daily_Usage_Hours'],
        'Mental_Health_Score': simulated_user['Mental_Health_Score']
    }])

    pred_sleep = model.predict(entrada)[0]
    actual_sleep = simulated_user['Sleep_Hours_Per_Night']

    print(f"🧠 Según tu edad ({simulated_user['Age']}), uso diario ({simulated_user['Avg_Daily_Usage_Hours']}h) y salud mental ({simulated_user['Mental_Health_Score']}),")
    print(f"🛌 deberías dormir aproximadamente **{pred_sleep:.2f} horas por noche**.")

    diff = actual_sleep - pred_sleep
    if abs(diff) < 0.5:
        print("✅ Estás durmiendo casi lo justo según tu perfil.")
    elif diff > 0:
        print(f"😴 Duermes **{abs(diff):.2f} horas más** de lo recomendado. ¡Bien si te sientes descansado!")
    else:
        print(f"⚠️ Duermes **{abs(diff):.2f} horas menos** de lo recomendado. Podrías considerar mejorar tu descanso.")


def logistic_regression_prediction():
    print("\n📊 Regresión Logística: ¿Tu uso de redes sociales podría afectar tu rendimiento académico?\n")

    df = load_data()
    df = clean_data(df)

    required_cols = ['Avg_Daily_Usage_Hours', 'Age', 'Sleep_Hours_Per_Night', 'Affects_Academic_Performance']
    if not all(col in df.columns for col in required_cols):
        print("❌ Columnas requeridas no encontradas.")
        return

    # Entrenamiento del modelo
    X = df[['Avg_Daily_Usage_Hours', 'Age', 'Sleep_Hours_Per_Night']]
    y = df['Affects_Academic_Performance']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Entrada del usuario
    entrada = pd.DataFrame([{
        'Avg_Daily_Usage_Hours': simulated_user['Avg_Daily_Usage_Hours'],
        'Age': simulated_user['Age'],
        'Sleep_Hours_Per_Night': simulated_user['Sleep_Hours_Per_Night']
    }])

    prob_afecta = model.predict_proba(entrada)[0][1]
    pred = model.predict(entrada)[0]

    # Mensaje principal
    print(f"🔍 Probabilidad estimada de que tu uso de redes sociales afecte tu rendimiento académico: **{prob_afecta:.2%}**")

    # Interpretación
    if prob_afecta >= 0.7:
        print("🚨 Es muy probable que tu uso de redes esté afectando tu desempeño académico.")
    elif 0.4 <= prob_afecta < 0.7:
        print("⚠️ Hay una probabilidad moderada. Tal vez deberías revisar tus hábitos digitales.")
    elif 0.2 <= prob_afecta < 0.4:
        print("🟡 La probabilidad es baja, pero no está de más tener cuidado con el tiempo en pantalla.")
    else:
        print("✅ Muy poco probable que tu uso esté afectando tu rendimiento académico. ¡Bien!")

    # Consejito según sus hábitos
    if simulated_user['Avg_Daily_Usage_Hours'] > 5:
        print("\n📵 Pasas más de 5 horas en redes al día. Tal vez puedas redistribuir ese tiempo para estudiar o descansar.")
    if simulated_user['Sleep_Hours_Per_Night'] < 6:
        print("💤 Dormir menos de 6 horas puede influir negativamente en tu concentración y rendimiento.")

    print()


def kmeans_clustering():
    print("\n🔎 K-Means: ¿Qué tipo de usuario eres según tus hábitos digitales?\n")

    df = load_data()
    df = clean_data(df)

    features = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score']
    if not all(col in df.columns for col in features):
        print("❌ Columnas requeridas no encontradas.")
        return

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df['Cluster'] = clusters

    # Centroides en valores originales
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids, columns=features)

    # Clasificar usuario
    entrada = pd.DataFrame([[
        simulated_user['Age'],
        simulated_user['Avg_Daily_Usage_Hours'],
        simulated_user['Sleep_Hours_Per_Night'],
        simulated_user['Mental_Health_Score']
    ]], columns=features)

    entrada_scaled = scaler.transform(entrada)
    cluster_usuario = kmeans.predict(entrada_scaled)[0]

    print(f"🧠 Tu perfil pertenece al grupo/clúster número: {cluster_usuario}\n")

    print("📌 Este grupo se caracteriza por:")
    for col in features:
        print(f"   • {col}: {centroids_df.loc[cluster_usuario, col]:.2f}")

    print("\n📊 Distribución de usuarios por grupo:")
    print(df['Cluster'].value_counts().sort_index())

    # Guardar gráfica 3D
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Colores
    colores = ['blue', 'green', 'orange']
    for i in range(3):
        grupo = df[df['Cluster'] == i]
        ax.scatter(grupo['Age'], grupo['Avg_Daily_Usage_Hours'], grupo['Mental_Health_Score'],
                   label=f'Cluster {i}', c=colores[i], s=40, alpha=0.6)

    # Punto del usuario
    ax.scatter(simulated_user['Age'], simulated_user['Avg_Daily_Usage_Hours'], simulated_user['Mental_Health_Score'],
               c='red', s=100, marker='X', label='Tú (usuario)', edgecolors='black')

    ax.set_xlabel('Edad')
    ax.set_ylabel('Uso Diario (hrs)')
    ax.set_zlabel('Salud Mental')

    ax.legend()
    ax.set_title("Agrupación de Usuarios según Hábitos Digitales")

    # Guardar en static/plots
    plots_dir = os.path.join(os.path.dirname(__file__), 'static', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    filename = "kmeans_clusters.png"
    graph_path = os.path.join(plots_dir, filename)
    plt.savefig(graph_path)
    plt.close()

    print(f"\n🖼️ Se ha generado una gráfica con tu posición en: static/plots/{filename}")

    # Interpretación amigable (ejemplo simple, puedes extenderlo)
    if cluster_usuario == 0:
        print("🔵 Eres un usuario equilibrado: buen sueño, uso moderado, buena salud mental.")
    elif cluster_usuario == 1:
        print("🟠 Tiendes a un uso elevado de redes y baja salud mental. ¡Atención a tus hábitos!")
    else:
        print("🟢 Pareces tener un perfil saludable, aunque podrías mejorar tus hábitos de sueño.")

    print("💡 Tip: Conoce tu grupo y ajusta tus hábitos para mantener el equilibrio digital.")


def correlation_analysis():
    print("\n📊 Diagnóstico Personalizado: ¿Tu rendimiento académico está en riesgo?\n")

    df = load_data()
    df = clean_data(df)

    if 'Affects_Academic_Performance' not in df.columns:
        print("❌ La columna 'Affects_Academic_Performance' no está presente.")
        return

    # Variables clave que se relacionan con el rendimiento
    factors = [
        'Avg_Daily_Usage_Hours',
        'Sleep_Hours_Per_Night',
        'Mental_Health_Score',
        'Addicted_Score',
        'Conflicts_Over_Social_Media'
    ]

    # Descripciones que el usuario entiende
    factor_descriptions = {
        'Avg_Daily_Usage_Hours': 'Horas promedio que usas redes sociales al día',
        'Sleep_Hours_Per_Night': 'Horas promedio que duermes cada noche',
        'Mental_Health_Score': 'Tu estado emocional (entre más alto, mejor salud mental)',
        'Addicted_Score': 'Nivel de adicción a redes sociales',
        'Conflicts_Over_Social_Media': 'Cantidad de conflictos por redes sociales'
    }

    # Evaluamos su situación personal comparando con el promedio
    print("🧪 Comparando tus datos con otros estudiantes similares:\n")
    riesgo = 0  # contador de riesgo

    for factor in factors:
        if factor in df.columns and factor in simulated_user:
            promedio = df[factor].mean()
            valor_usuario = simulated_user[factor]

            # Lógica de interpretación para el usuario
            if factor == 'Avg_Daily_Usage_Hours':
                if valor_usuario > promedio:
                    print(f"🔺 Usas redes sociales más de lo habitual ({valor_usuario:.1f}h frente a {promedio:.1f}h). Esto puede afectar tu concentración.")
                    riesgo += 1
                else:
                    print(f"✅ Tu tiempo en redes sociales está dentro de un rango saludable.")
            
            elif factor == 'Sleep_Hours_Per_Night':
                if valor_usuario < promedio:
                    print(f"🔻 Duermes menos de lo recomendado ({valor_usuario:.1f}h frente a {promedio:.1f}h). Esto puede impactar negativamente tu rendimiento.")
                    riesgo += 1
                else:
                    print(f"✅ Duermes una cantidad adecuada. ¡Sigue así!")

            elif factor == 'Mental_Health_Score':
                if valor_usuario < promedio:
                    print(f"🔻 Tu puntaje de salud mental es menor que el promedio. El estrés o ansiedad podrían influir en tu desempeño.")
                    riesgo += 1
                else:
                    print(f"✅ Tu salud mental parece estable comparada con otros estudiantes.")

            elif factor == 'Addicted_Score':
                if valor_usuario > promedio:
                    print(f"🔺 Tienes un nivel de adicción más alto que la mayoría. Esto podría afectar tu enfoque académico.")
                    riesgo += 1
                else:
                    print(f"✅ Tu nivel de adicción a redes parece controlado.")

            elif factor == 'Conflicts_Over_Social_Media':
                if valor_usuario > promedio:
                    print(f"🔺 Has reportado más conflictos por redes sociales que otros. Esto puede generar distracciones emocionales.")
                    riesgo += 1
                else:
                    print(f"✅ Muestras buena gestión de tus relaciones respecto al uso de redes sociales.")

    # Resultado general
    print("\n📌 Resultado final del diagnóstico:")
    if riesgo >= 4:
        print("❗ Con base en tus datos, parece que varios factores pueden estar afectando tu rendimiento académico.")
        print("🧠 Considera mejorar tus hábitos digitales, tu descanso y salud emocional.")
    elif riesgo >= 2:
        print("⚠️ Algunos factores podrían estar influyendo en tu desempeño. Podrías beneficiarte con pequeños cambios.")
    else:
        print("✅ En general, tu perfil muestra pocos indicadores de riesgo. ¡Sigue cuidando tu bienestar académico!")

    print("\n🔎 Este diagnóstico se basa en patrones observados en estudiantes reales y no sustituye una evaluación profesional.")


def academic_level_distribution():
    """
    Clasifica a los usuarios por nivel académico y muestra el porcentaje
    de cada categoría, además de informar al usuario simulado su posición.
    """
    print("\n🎓 Distribución por Nivel Académico\n")

    # Cargar y limpiar datos
    df = load_data()
    df = clean_data(df)

    # Verificar columna
    if 'Academic_Level' not in df.columns:
        print("❌ La columna 'Academic_Level' no está presente en el dataset.")
        return

    # Cálculo de porcentajes
    counts = df['Academic_Level'].value_counts(normalize=True) * 100
    counts = counts.round(1)  # un decimal

    # Mostrar la distribución completa
    print("📊 Porcentaje de usuarios por nivel académico:")
    for level, pct in counts.items():
        print(f" - {level}: {pct}%")

    # Determinar nivel del usuario simulado
    user_level = simulated_user.get('Academic_Level')
    if user_level in counts:
        user_pct = counts[user_level]
        print(f"\n👤 Perteneces al {user_pct}% de usuarios que cursan {user_level}.")
    else:
        print(f"\n❓ Tu nivel académico '{user_level}' no se encuentra en los datos.")


def sleep_comparison_by_profile():
    print("\n💤 Análisis del Sueño: ¿Duermes lo suficiente según tu perfil?\n")

    df = load_data()
    df = clean_data(df)

    if 'Age' not in df.columns or 'Sleep_Hours_Per_Night' not in df.columns or 'Academic_Level' not in df.columns:
        print("❌ Columnas requeridas no encontradas.")
        return

    user_age = simulated_user['Age']
    user_level = simulated_user['Academic_Level']
    user_sleep = simulated_user['Sleep_Hours_Per_Night']

    # Crear grupos por edad similar (±2 años) y mismo nivel académico
    df_group = df[
        (df['Age'].between(user_age - 2, user_age + 2)) &
        (df['Academic_Level'] == user_level)
    ]

    if df_group.empty:
        print("❗ No se encontraron suficientes usuarios con perfil similar para comparar.")
        return

    avg_sleep = df_group['Sleep_Hours_Per_Night'].mean()

    print(f"🧑‍🎓 Usuarios similares (Edad: {user_age} ±2 años, Nivel: {user_level}) duermen en promedio: {avg_sleep:.1f} horas por noche.")
    print(f"😴 Tú reportaste: {user_sleep:.1f} horas por noche.\n")

    if user_sleep >= avg_sleep + 1:
        print("✅ Duermes más que el promedio. ¡Sigue así!")
    elif user_sleep <= avg_sleep - 1:
        print("⚠️ Duermes menos que el promedio de tu grupo. Intenta mejorar tu descanso.")
    else:
        print("👌 Duermes aproximadamente lo mismo que otros en tu perfil.")


def predict_academic_performance():
    print("\n📚 Predicción: ¿Tu estilo de vida afecta tu rendimiento académico?\n")

    df = load_data()
    df = clean_data(df)

    required_cols = [
        'Avg_Daily_Usage_Hours', 'Age', 'Sleep_Hours_Per_Night',
        'Mental_Health_Score', 'Affects_Academic_Performance'
    ]

    if not all(col in df.columns for col in required_cols):
        print("❌ Columnas requeridas no encontradas.")
        return

    X = df[['Avg_Daily_Usage_Hours', 'Age', 'Sleep_Hours_Per_Night', 'Mental_Health_Score']]
    y = df['Affects_Academic_Performance']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    entrada = pd.DataFrame([{
        'Avg_Daily_Usage_Hours': simulated_user['Avg_Daily_Usage_Hours'],
        'Age': simulated_user['Age'],
        'Sleep_Hours_Per_Night': simulated_user['Sleep_Hours_Per_Night'],
        'Mental_Health_Score': simulated_user['Mental_Health_Score']
    }])

    prob = model.predict_proba(entrada)[0][1]
    pred = model.predict(entrada)[0]

    print(f"🧠 Basado en tus hábitos de uso, edad, sueño y salud mental:")

    if pred == 1:
        print(f"⚠️ Hay una **alta probabilidad ({prob:.2%})** de que tu rendimiento académico se vea afectado.")
        print("📝 Considera mejorar tus hábitos de sueño o reducir el tiempo en redes sociales.")
    else:
        print(f"✅ Es poco probable ({(1 - prob):.2%}) que tu rendimiento académico esté en riesgo.")
        print("🎉 ¡Buen trabajo manteniendo un equilibrio saludable!")


def platform_usage_distribution():
    print("\n📱 Análisis de plataforma más usada entre los estudiantes\n")

    df = load_data()
    df = clean_data(df)

    if 'Most_Used_Platform' not in df.columns:
        print("❌ La columna 'Most_Used_Platform' no está presente en el dataset.")
        return

    # Contar las plataformas
    platform_counts = df['Most_Used_Platform'].value_counts()
    total = platform_counts.sum()
    platform_percentages = (platform_counts / total * 100).round(2)

    # Obtener las 3 más populares
    top_3 = platform_percentages.head(3)
    top_platforms = top_3.index.tolist()

    # Preparar la gráfica
    plt.figure(figsize=(10, 6))
    bars = plt.bar(platform_percentages.index, platform_percentages.values, color='lightgray')

    # Resaltar la del usuario si existe
    user_platform = simulated_user['Most_Used_Platform']
    if user_platform in platform_percentages:
        idx = platform_percentages.index.tolist().index(user_platform)
        bars[idx].set_color('orange')
        plt.text(idx, platform_percentages[user_platform] + 1, "👤 Tú", ha='center', color='orange', fontweight='bold')

    # Etiquetas
    plt.title("Distribución de plataformas más usadas por los estudiantes")
    plt.ylabel("Porcentaje de usuarios (%)")
    plt.xlabel("Plataformas")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Guardar la imagen
    plots_dir = os.path.join(os.path.dirname(__file__), 'static', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    graph_path = os.path.join(plots_dir, 'platform_distribution.png')
    plt.savefig(graph_path)
    plt.close()

    # Mostrar resultado al usuario
    print("📊 Porcentaje de uso por plataforma:")
    for platform, pct in platform_percentages.items():
        print(f" - {platform}: {pct}%")

    print("\n📍 Tu plataforma principal es:", user_platform)
    if user_platform in top_platforms:
        rank = top_platforms.index(user_platform) + 1
        print(f"✅ Estás en el TOP {rank} de plataformas más utilizadas.")
    else:
        print(f"ℹ️ Tu plataforma no se encuentra en el TOP 3, pero igual es representativa en el análisis.")

    print(f"\n🖼️ Gráfica generada en: static/plots/platform_distribution.png")

