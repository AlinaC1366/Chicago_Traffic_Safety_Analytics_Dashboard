import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import statsmodels.api as sm

st.set_page_config(page_title="Siguranță Rutieră Chicago", layout="wide")
st.title("🚨 Analiza Siguranței Rutiere: Chicago")
st.markdown("---")

@st.cache_data
def incarca_date():
    df = pd.read_csv("chicago_accidents.csv")

    df['injuries_total'] = pd.to_numeric(df['injuries_total'], errors='coerce')
    df['posted_speed_limit'] = pd.to_numeric(df['posted_speed_limit'], errors='coerce')
    df['num_units'] = pd.to_numeric(df['num_units'], errors='coerce')

    # 1. VALORI LIPSĂ
    df['injuries_total'] = df['injuries_total'].fillna(0)

    # 2. VALORI EXTREME (Outliers)
    df = df[(df['posted_speed_limit'] >= 5) & (df['posted_speed_limit'] <= 80)]
    df = df.dropna(subset=['weather_condition', 'longitude', 'latitude', 'num_units'])

    return df


df = incarca_date()

st.sidebar.header("Filtrează Accidentele")
viteza_maxima = st.sidebar.slider("Viteza maximă în zonă (mph)",
                                  int(df['posted_speed_limit'].min()),
                                  int(df['posted_speed_limit'].max()),
                                  60)

df_filtrat = df[df['posted_speed_limit'] <= viteza_maxima]

# Buton Download (Bifăm interactivitatea maximă)
csv_curatat = df_filtrat.to_csv(index=False).encode('utf-8')
st.sidebar.markdown("---")
st.sidebar.download_button(label="📥 Descarcă Datele Filtrate", data=csv_curatat, file_name='chicago_filtrat.csv',
                           mime='text/csv')

col1, col2, col3 = st.columns(3)
col1.metric("Total Accidente", len(df_filtrat))
col2.metric("Victime Raportate", int(df_filtrat['injuries_total'].sum()))
col3.metric("Vehicule Implicate (Mediu)", f"{df_filtrat['num_units'].mean():.1f}")
st.markdown("---")


gdf = gpd.GeoDataFrame(df_filtrat, geometry=gpd.points_from_xy(df_filtrat.longitude, df_filtrat.latitude))
gdf.crs = "EPSG:4326"

centru_chicago = Point(-87.6233, 41.8827)
centru_geo = gpd.GeoSeries([centru_chicago], crs="EPSG:4326")

centru_metric = centru_geo.to_crs("EPSG:3857").iloc[0]
gdf['distanta_centru_km'] = gdf.to_crs("EPSG:3857").distance(centru_metric) / 1000

tab1, tab2, tab3, tab4 = st.tabs(
    ["📍 Harta Geografică", "📊 Factori de Risc", "🤖 Inteligență Artificială", "📈 Statsmodels (Regresie)"])

with tab1:
    # CERINȚA 2: GEOPANDAS
    st.header("📍 Harta Accidentelor și Distanța")
    st.write(
        f"Distanța medie a accidentelor față de centrul orașului este de **{gdf['distanta_centru_km'].mean():.1f} km**.")
    st.map(gdf)

with tab2:
    # CERINȚA 4 & 5: Grupare și Agregare
    st.header("📊 Factori de Risc (Luminozitate)")
    stats_lumina = df_filtrat.groupby('lighting_condition').agg({
        'crash_record_id': 'count',
        'injuries_total': 'sum'
    }).rename(columns={'crash_record_id': 'total_accidente', 'injuries_total': 'total_raniți'})

    stats_lumina = stats_lumina.sort_values('total_accidente', ascending=False)
    st.bar_chart(stats_lumina['total_accidente'])
    st.table(stats_lumina)

with tab3:
    # CERINȚA 6, 7 & 8: Scikit-Learn
    st.header("🤖 Inteligență Artificială: Zonele de Risc")
    df_ml = gdf.copy().dropna(subset=['weather_condition'])

    le = LabelEncoder()
    df_ml['meteo_codificat'] = le.fit_transform(df_ml['weather_condition'])

    scaler = StandardScaler()
    df_ml[['viteza_scalata', 'distanta_scalata']] = scaler.fit_transform(
        df_ml[['posted_speed_limit', 'distanta_centru_km']])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_ml['ID_Cluster'] = kmeans.fit_predict(df_ml[['viteza_scalata', 'distanta_scalata']])

    nume_clastere = {0: "Risc Mediu (Oraș)", 1: "Risc Critic (Viteză)", 2: "Risc Scăzut (Centru Aglomerat)"}
    df_ml['Clasificare_AI'] = df_ml['ID_Cluster'].map(nume_clastere)

    fig_cluster = px.scatter(df_ml, x="distanta_centru_km", y="posted_speed_limit", color="Clasificare_AI",
                             title="Cum grupează AI-ul accidentele",
                             labels={"distanta_centru_km": "Distanță față de Centru (km)",
                                     "posted_speed_limit": "Viteză Legală (mph)"})
    st.plotly_chart(fig_cluster, width='stretch')

with tab4:
    st.header("📈 Model Statistic: Ce provoacă victimele?")
    st.write(
        "Prin regresia multiplă, măsurăm matematic dacă Viteza și Numărul de vehicule sunt direct responsabile pentru numărul de răniți.")

    # Pregătim datele pentru model (fără valori lipsă)
    df_reg = df_filtrat.dropna(subset=['posted_speed_limit', 'num_units', 'injuries_total']).copy()

    # Definirea variabilelor
    X = df_reg[['posted_speed_limit', 'num_units']]
    X = sm.add_constant(X, has_constant='add')
    y = df_reg['injuries_total']

    # Antrenarea modelului
    model = sm.OLS(y, X).fit()

    # Afișarea tabelului cu rezultate
    tabel_date = model.summary().tables[1].data
    df_rezultate = pd.DataFrame(tabel_date[1:], columns=tabel_date[0])
    st.dataframe(df_rezultate)

    st.markdown("---")
    st.subheader("🔮 Simulatorul Poliției: Estimează Răniții")
    col_sim1, col_sim2 = st.columns(2)
    viteza_sim = col_sim1.number_input("Limita de viteză în zonă (mph)", 10, 80, 40)
    masini_sim = col_sim2.number_input("Câte mașini s-au ciocnit?", 1, 10, 2)

    if st.button("Simulează (Apasă aici)"):
        # Creăm un mini-tabel pentru predicție
        date_predictie = pd.DataFrame({'const': [1], 'posted_speed_limit': [viteza_sim], 'num_units': [masini_sim]})
        predictie = max(0, model.predict(date_predictie)[0])
        st.error(f"🚑 Modelul estimează statistic o medie de **{predictie:.2f}** victime la acest impact.")

st.markdown("---")
with st.expander("🛠️ Detalii Tehnice"):
    st.write("1. **Streamlit:** Interfață, filtre slider, metrici, tab-uri, butoane.")
    st.write("2. **Geopandas:** Transformare în obiecte spațiale și re-proiecție (EPSG:3857) pentru distanță KM.")
    st.write("3. **Tratare:** `fillna` pentru lipsa victimelor și filtre logice pentru eliminarea erorilor de viteză.")
    st.write("4. **Pandas Groupby & Agg:** Gruparea datelor pe starea luminii stradale.")
    st.write("5. **Funcții de grup:** `sum` și `count` pentru a găsi mediile victimelor.")
    st.write("6. **Codificare:** `LabelEncoder` transformă textul condițiilor meteo în cifre.")
    st.write("7. **Scalare:** `StandardScaler` aduce viteza și distanța pe aceeași axă matematică.")
    st.write("8. **Scikit-Learn:** `K-Means` împarte automat accidentele în zone de Risc.")
    st.write("9. **Statsmodels:** Regresie OLS care corelează statistic viteza și nr. de mașini cu numărul de victime.")
    st.latex(r"d(P_1, P_2) = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}")