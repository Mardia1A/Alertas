import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="D", layout="wide")

st.markdown("""
<style>
    h1, h2, h3, .stText, .stMarkdown {
        color: #004d40;
    }
    .css-10trblm { font-size: 16px; }
    .stPlotlyChart, .stImage, .stAltText {
        max-height: 300px !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Dashboard de Decisiones Clínicas")

st.markdown("""
### ❓ Pregunta de investigación:
¿Cómo generar alertas tempranas para evitar complicaciones en pacientes con insuficiencia cardíaca?

### 🎯 Objetivo:
Desarrollar un sistema de alertas que informe al médico sobre variables de riesgo y permita a los estudiantes explorar cómo los cambios en estas variables afectan el desenlace clínico.

---
""")

# Cargar datos
df = pd.read_csv("historiales_clinicos.csv")

# 1. Histogramas de variables relevantes
st.subheader("1. Variables clínicas relevantes")
opciones_hist = st.multiselect("Selecciona variables para ver histogramas:",
                                ["serum_creatinine", "ejection_fraction", "serum_sodium"],
                                default=["serum_creatinine"])
colores = {"serum_creatinine": "#2e7d32", "ejection_fraction": "#0288d1", "serum_sodium": "#00796b"}

for var in opciones_hist:
    fig, ax = plt.subplots(figsize=(3, 1))
    sns.histplot(df[var], bins=30, kde=True, color=colores.get(var, "gray"), ax=ax)
    ax.set_title(f"{var.replace('_', ' ').title()}", fontsize=9)
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    st.pyplot(fig)

# 2. Outliers en variables clínicas
st.subheader("2. Outliers en pacientes críticos")
variables_out = st.multiselect("Selecciona variables para analizar outliers:",
                               ["platelets", "serum_creatinine", "ejection_fraction"],
                               default=["platelets"])

titulos_out = {
    "platelets": "Plaquetas",
    "serum_creatinine": "Creatinina",
    "ejection_fraction": "Fracción de Eyección"
}

for col in variables_out:
    fig, ax = plt.subplots(figsize=(3, 1))
    sns.boxplot(y=df[col], ax=ax, color="#4db6ac")
    ax.set_title(titulos_out[col], fontsize=9)
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("""
Se identificaron valores extremos (outliers) en variables como creatinina, plaquetas y fosfoquinasa.  
Se conservaron debido al estado crítico de los pacientes, ya que estos valores extremos pueden indicar gravedad clínica.  
También se evidencian outliers relevantes en fracción de eyección.

---
""")

# 3. Análisis de dispersión vs muerte
st.subheader("3. Análisis de dispersión: Variables clínicas vs Muerte")
opciones_disp = st.multiselect("Selecciona gráfico de dispersión:",
                                ["Edad", "Creatinina", "Eyección", "Tiempo"],
                                default=["Edad"])

map_vars = {
    "Edad": "age",
    "Creatinina": "serum_creatinine",
    "Eyección": "ejection_fraction",
    "Tiempo": "time"
}

for op in opciones_disp:
    fig, ax = plt.subplots(figsize=(3, 1))
    sns.scatterplot(x=df[map_vars[op]], y=df["DEATH_EVENT"], ax=ax, color="#43a047")
    ax.set_title(f"{op} vs Muerte", fontsize=9)
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("""
**Edad vs Muerte**: mayor concentración de muertes desde los 60 años, aunque también hay casos en jóvenes.  

**Creatinina Sérica vs Muerte**: muertes frecuentes cuando supera 1.5 mg/dL, muy notorias desde 2.0 mg/dL.  

**Fracción de Eyección vs Muerte**: valores menores al 40% están fuertemente asociados con fallecimientos.  

**Tiempo de seguimiento vs Muerte**: quienes mueren tienden a tener menor tiempo, reflejando urgencia clínica.

---
""")

# 4. Modelo seleccionado
st.subheader("4. Evolución del modelo y decisión final")
st.markdown("""
Inicialmente se evaluó un árbol de decisión incluyendo la variable **tiempo**, que arrojó:  
- Accuracy: 0.75  
- F1-score (muertos): 0.52  

Pero se decidió **remover el tiempo** del modelo, ya que el objetivo no era predecir muerte sino generar **alertas tempranas**.  

Se probaron modelos como regresión logística y redes neuronales, pero no mejoraron significativamente.  

Se optó por **Random Forest**, realizando tuning de hiperparámetros con RandomizedSearchCV:  
- Mejores parámetros: `max_depth=11`, `min_samples_split=9`, `n_estimators=178`  
- Accuracy: 0.75  
- F1-score (muertos): 0.57  

Finalmente, se seleccionó este modelo por su equilibrio entre sensibilidad clínica y rendimiento.

---
""")

# 5. Clustering como herramienta exploratoria
st.subheader("5. Agrupamiento de perfiles clínicos")
st.markdown("""
Se aplicó **K-Means Clustering** para encontrar perfiles similares de pacientes.  
Se usaron las variables normalizadas de creatinina, eyección y plaquetas.
""")

scaler = StandardScaler()
x_cluster = scaler.fit_transform(df[["serum_creatinine", "ejection_fraction", "platelets"]])
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(x_cluster)

fig, ax = plt.subplots(figsize=(3, 1))
cluster_counts = df["cluster"].value_counts().sort_index()
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="Set2", ax=ax)
ax.set_title("Distribución de Clusters", fontsize=9)
ax.set_xlabel("Cluster", fontsize=8)
ax.set_ylabel("Número de Pacientes", fontsize=8)
ax.tick_params(labelsize=6)
plt.tight_layout()
st.pyplot(fig)

st.markdown("""
Cada grupo sugiere un tipo de perfil clínico que podría ser priorizado en el sistema de alertas o revisiones médicas.  
Esto permite visualizar **subgrupos de pacientes con condiciones similares** para intervenciones específicas.
""")
