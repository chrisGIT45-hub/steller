import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import base64
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.graph_objects as go
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Steller Intelligence",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. HELPER FUNCTIONS & STYLING ---
@st.cache_data
def get_img_as_base64(file):
    """Encodes a local image file to a Base64 string."""
    if os.path.exists(file):
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    st.warning(f"Background image '{file}' not found. Using fallback color.")
    return None

def apply_custom_style():
    """Applies custom CSS for styling, including the background image."""
    img_base64 = get_img_as_base64("bg1.jpg")
    
    # Use background image if found, otherwise use a fallback color
    background_style = f"background-image: url(data:image/jpg;base64,{img_base64});" if img_base64 else "background-color: #0c0e18;"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400&display=swap');
        
        /* Main App Styling with Background Image */
        .stApp {{
            {background_style}
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Centered header with reduced size */
        .header-container {{
            padding: 4rem 1rem; /* Vertical and horizontal padding */
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            color: white;
        }}

        .header-container h1 {{
            font-family: 'Orbitron', sans-serif;
            font-size: 3.5rem;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
        }}
        .header-container h3 {{
            font-family: 'Roboto', sans-serif;
            max-width: 700px;
            font-size: 1.2rem;
            text-shadow: 1px 1px 4px rgba(0,0,0,0.7);
        }}

        /* Styling for other UI elements for better visibility */
        .stButton>button {{ border: 2px solid #00A2FF; background-color: rgba(12, 14, 24, 0.6); color: #00A2FF; }}
        .stButton>button:hover {{ background-color: #00A2FF; color: #0c0e18; }}
        .stExpander, .stMetric, .stSelectbox > div, .stForm {{
            background-color: rgba(22, 27, 34, 0.85);
            border: 1px solid #30363d;
            border-radius: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- 3. DATA LOADING & MODEL TRAINING (CACHED) ---
@st.cache_resource
def load_models_and_data(zip_file_path):
    """Loads data from a zip file, preprocesses it, and trains all necessary models."""
    try:
        csv_file_name = 'planetary_system.csv'
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            with z.open(csv_file_name) as f:
                df = pd.read_csv(f)
    except FileNotFoundError:
        st.error(f"Error: The data file '{zip_file_path}' was not found.")
        return None
    except KeyError:
        st.error(f"Error: Could not find '{csv_file_name}' inside the zip archive.")
        return None

    # --- Model 1: Planet Classification ---
    df_class = df.copy()
    conditions = [
        (df_class['pl_radj'] > 6) & (df_class['pl_bmassj'] > 0.5),
        (df_class['pl_radj'] >= 2) & (df_class['pl_radj'] <= 6),
        (df_class['pl_rade'] > 1.25) & (df_class['pl_rade'] < 2),
        (df_class['pl_rade'] <= 1.25) & (df_class['pl_bmasse'] < 10)
    ]
    choices = ['Gas Giant', 'Neptune-like', 'Super-Earth', 'Terrestrial']
    df_class['planet_type'] = np.select(conditions, choices, default='Unknown')
    df_class.dropna(subset=['pl_orbper','st_mass','pl_dens','st_rad','st_teff', 'planet_type'], inplace=True)
    df_class = df_class[df_class['planet_type'].isin(['Neptune-like', 'Super-Earth', 'Terrestrial'])]
    X_class = df_class[['pl_orbper','st_mass','pl_dens','st_rad','st_teff']]
    y_class = df_class['planet_type']
    model_classify = RandomForestClassifier(random_state=42).fit(X_class, y_class)

    # --- Model 2: Optimal Discovery Method ---
    df_disc = df.copy()
    features_disc = ['sy_snum', 'sy_pnum', 'sy_mnum', 'sy_dist', 'st_mass', 'st_rad', 'st_lum', 'st_logg', 'st_age', 'st_dens', 'st_met']
    df_disc.dropna(subset=[ 'discoverymethod'] + features_disc, inplace=True)
    top_methods = df_disc['discoverymethod'].value_counts().nlargest(4).index
    df_disc = df_disc[df_disc['discoverymethod'].isin(top_methods)]
    X_disc = df_disc[features_disc]
    y_disc = df_disc['discoverymethod']
    label_encoder_disc = LabelEncoder().fit(y_disc)
    y_encoded_disc = label_encoder_disc.transform(y_disc)
    scaler_disc = StandardScaler().fit(X_disc)
    X_scaled_disc = scaler_disc.transform(X_disc)
    model_discovery = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_scaled_disc, y_encoded_disc)
    
    # --- Model 4: Clustering for Archetypes ---
    df_cluster = df.copy()
    features_cluster = ['pl_orbsmax','pl_radj','pl_bmassj','pl_dens','pl_eqt','st_teff','st_met']
    df_cluster.dropna(subset=features_cluster, inplace=True)
    X_scaled_cluster = StandardScaler().fit_transform(df_cluster[features_cluster])
    X_pca = PCA(n_components=2).fit_transform(X_scaled_cluster)
    labels_pca = DBSCAN(eps=1.9, min_samples=5).fit_predict(X_pca)
    df_cluster['pca_1'] = X_pca[:, 0]
    df_cluster['pca_2'] = X_pca[:, 1]
    df_cluster['archetype'] = labels_pca
    df_cluster = df_cluster[df_cluster['archetype'] != -1]

    # --- Model 5: Controversial Planet Prediction ---
    df_controv = df.copy()
    features_controv = ['pl_trandep', 'pl_rvamp', 'pl_radjerr1', 'pl_massjerr1', 'sy_snum', 'sy_pnum', 'discoverymethod', 'disc_facility']
    numerical_features = ['pl_trandep', 'pl_rvamp', 'pl_radjerr1', 'pl_massjerr1', 'sy_snum', 'sy_pnum']
    categorical_features = ['discoverymethod', 'disc_facility']
    
    num_imputer = SimpleImputer(strategy='median').fit(df_controv[numerical_features])
    df_controv[numerical_features] = num_imputer.transform(df_controv[numerical_features])
    
    cat_imputer = SimpleImputer(strategy='most_frequent').fit(df_controv[categorical_features])
    df_controv[categorical_features] = cat_imputer.transform(df_controv[categorical_features])
    
    encoders_controv = {col: LabelEncoder().fit(df_controv[col]) for col in categorical_features}
    for col in categorical_features:
        df_controv[col] = encoders_controv[col].transform(df_controv[col])
        
    X_controv = df_controv[features_controv]
    y_controv = df_controv['pl_controv_flag']
    model_controversial = RandomForestClassifier(random_state=42, class_weight='balanced').fit(X_controv, y_controv)
    
    feature_importances = pd.DataFrame({
        'feature': X_controv.columns, 'importance': model_controversial.feature_importances_
    }).sort_values('importance', ascending=False)

    return (df, model_classify, model_discovery, scaler_disc, label_encoder_disc, 
            df_cluster, model_controversial, feature_importances, 
            num_imputer, cat_imputer, encoders_controv, numerical_features, categorical_features)

def calculate_habitable_zone(star_luminosity):
    inner_boundary = np.sqrt(star_luminosity / 1.1)
    outer_boundary = np.sqrt(star_luminosity / 0.53)
    return inner_boundary, outer_boundary

def plot_habitable_zone(star_lum, planet_orbit_au, planet_name):
    hz_inner, hz_outer = calculate_habitable_zone(star_lum)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[hz_inner, hz_outer, hz_outer, hz_inner], y=[-1, -1, 1, 1], fill="toself", fillcolor='rgba(0, 255, 0, 0.3)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="text", text=f"Habitable Zone<br>{hz_inner:.2f} - {hz_outer:.2f} AU", name='Habitable Zone'))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(color='yellow', size=20, symbol='star'), name='Host Star', hoverinfo="text", text="Host Star"))
    fig.add_trace(go.Scatter(x=[planet_orbit_au], y=[0], mode='markers', marker=dict(color='#00A2FF', size=12), name=planet_name, hoverinfo="text", text=f"{planet_name}<br>Orbit: {planet_orbit_au:.2f} AU"))
    fig.update_layout(title="Planet's Position Relative to Habitable Zone", xaxis_title="Distance from Star (AU)", yaxis_visible=False, plot_bgcolor='rgba(12,14,24,0.8)', paper_bgcolor='rgba(0,0,0,0)', font_color='#E0E0E0', showlegend=False)
    return fig


# --- 4. MAIN APP LAYOUT ---

apply_custom_style()

# --- HEADER SECTION ---
with st.container():
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    st.title("STELLER INTELLIGENCE")
    st.markdown("### Analyze, classify, and explore exoplanets using machine learning. Predict discovery confidence and uncover new celestial archetypes.")
    st.markdown('</div>', unsafe_allow_html=True)

# Load data and models
loaded_data = load_models_and_data('planetary_system.zip')

# Stop the app if data loading fails
if loaded_data is None:
    st.stop()

(df_main, model_classify, model_discovery, scaler_disc, label_encoder_disc, 
 df_cluster, model_controversial, feature_importances, 
 num_imputer, cat_imputer, encoders_controv, numerical_features, categorical_features) = loaded_data

st.markdown("---")

# --- TARGET SELECTION ---
st.header("1. Target Selection")
st.write("Begin by selecting a known exoplanet from the archive or define a hypothetical one for analysis.")
col1, col2 = st.columns([2, 1])
with col1:
    planet_options = list(df_main['pl_name'].unique())
    planet_options.insert(0, "Select a Planet...")
    selected_planet_name = st.selectbox("Select a Planet from the NASA Exoplanet Archive:", options=planet_options, label_visibility="collapsed")
with col2:
    st.write("") # for vertical alignment
    st.write("")
    if st.button("Analyze a Hypothetical Planet"):
        st.session_state.show_hypothetical_form = not st.session_state.get('show_hypothetical_form', False)

target_data = None
is_hypothetical = False

if 'show_hypothetical_form' not in st.session_state:
    st.session_state.show_hypothetical_form = False

if st.session_state.get('show_hypothetical_form', False):
    with st.form("hypothetical_form"):
        st.subheader("Define Hypothetical Planet Parameters")
        st.info("Enter the characteristics of your planet and its star. The models will predict its nature based on these inputs.")
        c1, c2, c3 = st.columns(3)
        pl_orbper = c1.number_input("Orbital Period (days)", min_value=0.1, value=365.25, step=1.0)
        st_mass = c2.number_input("Stellar Mass (Solar Masses)", min_value=0.1, value=1.0)
        pl_dens = c3.number_input("Planet Density (g/cm³)", min_value=0.1, value=5.51)
        st_rad = c1.number_input("Stellar Radius (Solar Radii)", min_value=0.1, value=1.0)
        st_teff = c2.number_input("Stellar Temperature (K)", min_value=2000, value=5778)
        sy_snum = c3.number_input("Number of Stars in System", min_value=1, value=1, step=1)
        sy_pnum = c1.number_input("Number of Planets in System", min_value=1, value=1, step=1)
        sy_dist = c2.number_input("Distance from Earth (parsecs)", min_value=1.0, value=50.0)
        st_lum = c3.number_input("Stellar Luminosity (log(Solar))", min_value=-5.0, value=0.0)
        pl_orbsmax = c1.number_input("Planet Orbit Semi-Major Axis (AU)", min_value=0.01, value=1.0)
        
        submitted = st.form_submit_button("Analyze This Planet")
        if submitted:
            hypothetical_dict = {'pl_name': 'Hypothetical Planet','pl_orbper': [pl_orbper],'st_mass': [st_mass],'pl_dens': [pl_dens],'st_rad': [st_rad],'st_teff': [st_teff],'sy_snum': [sy_snum],'sy_pnum': [sy_pnum],'sy_mnum': [0],'sy_dist': [sy_dist],'st_lum': [st_lum],'st_logg': [4.44],'st_age': [4.6],'st_met': [0.0],'pl_trandep': [1.0],'pl_rvamp': [1.0],'pl_radjerr1': [0.01],'pl_massjerr1': [0.01],'discoverymethod': ['Transit'],'disc_facility': ['Kepler'],'pl_controv_flag': [0],'pl_orbsmax': [pl_orbsmax],'pl_radj': [0.1],'pl_bmassj': [0.01],'pl_eqt': [288]}
            target_data = pd.DataFrame(hypothetical_dict)
            is_hypothetical = True
            st.session_state.show_hypothetical_form = False
            selected_planet_name = "Hypothetical Planet"

if selected_planet_name not in ["Select a Planet...", "Hypothetical Planet"]:
    target_data = df_main[df_main['pl_name'] == selected_planet_name].iloc[[0]]

if target_data is not None:
    st.success(f"Analysis loaded for: **{selected_planet_name}**")
    st.markdown("---")
    col_profile, col_discovery = st.columns(2)
    with col_profile:
        st.header("2. Planetary Profile")
        class_features = ['pl_orbper', 'st_mass', 'pl_dens', 'st_rad', 'st_teff']
        planet_class_pred = model_classify.predict(target_data[class_features])[0]
        st.metric(label="Predicted Classification", value=planet_class_pred)
        with st.expander("View Classification Probabilities"):
            class_probs = model_classify.predict_proba(target_data[class_features])[0]
            prob_df = pd.DataFrame(class_probs, index=model_classify.classes_, columns=['Probability'])
            st.bar_chart(prob_df.sort_values('Probability', ascending=False))
        density_val = target_data['pl_dens'].iloc[0]
        st.metric(label="Planet Density", value=f"{density_val:.2f} g/cm³")
    with col_discovery:
        st.header("3. Discovery & Analysis")
        if is_hypothetical:
             st.info("Discovery Confidence is not applicable for hypothetical planets.")
        else:
            temp_data_controv = target_data.copy()
            temp_data_controv[numerical_features] = num_imputer.transform(temp_data_controv[numerical_features])
            for col in categorical_features:
                 temp_data_controv[col] = encoders_controv[col].transform(temp_data_controv[col])
            confidence_prob = model_controversial.predict_proba(temp_data_controv[features_controv])[0]
            confidence_score = confidence_prob[0] * 100
            st.metric(label="Discovery Confidence Score", value=f"{confidence_score:.1f}%")
            with st.expander("Show Key Confidence Factors"):
                st.dataframe(feature_importances.head())
        
        disc_features_cols = ['sy_snum', 'sy_pnum', 'sy_mnum', 'sy_dist', 'st_mass', 'st_rad', 'st_lum', 'st_logg', 'st_age', 'st_dens', 'st_met']
        scaled_disc_features = scaler_disc.transform(target_data[disc_features_cols])
        method_pred = label_encoder_disc.inverse_transform(model_discovery.predict(scaled_disc_features))[0]
        st.metric(label="Optimal Search Method", value=method_pred)

    st.markdown("---")
    st.header("4. Habitability & Cosmic Context")
    if pd.notna(target_data['st_lum'].iloc[0]) and pd.notna(target_data['pl_orbsmax'].iloc[0]):
        if st.toggle("Show Habitable Zone Visualization", value=True):
            hz_plot = plot_habitable_zone(10**target_data['st_lum'].iloc[0], target_data['pl_orbsmax'].iloc[0], selected_planet_name)
            st.plotly_chart(hz_plot, use_container_width=True)

st.markdown("---")
st.header("Planet Archetype Explorer")
st.write("Explore clusters of known exoplanets based on their physical properties. Hover over a point to see its name. The selected planet is highlighted in white.")
plot_df = df_cluster[['pl_name', 'pca_1', 'pca_2', 'archetype']].copy()
fig_cluster = px.scatter(plot_df, x='pca_1', y='pca_2', color='archetype', hover_name='pl_name', color_continuous_scale=px.colors.sequential.Plasma, title='DBSCAN Clustering of Planets after PCA')
if target_data is not None and not is_hypothetical:
    selected_planet_cluster_info = df_cluster[df_cluster['pl_name'] == selected_planet_name]
    if not selected_planet_cluster_info.empty:
        fig_cluster.add_trace(go.Scatter(x=selected_planet_cluster_info['pca_1'], y=selected_planet_cluster_info['pca_2'], mode='markers', marker=dict(color='white', size=12, symbol='star', line=dict(color='black', width=1)), name=f'Selected: {selected_planet_name}'))
fig_cluster.update_layout(xaxis_title='Principal Component 1', yaxis_title='Principal Component 2', plot_bgcolor='rgba(12,14,24,0.8)', paper_bgcolor='rgba(0,0,0,0)', font_color='#E0E0E0', legend_title_text='Archetype')
st.plotly_chart(fig_cluster, use_container_width=True)

st.markdown("---")
st.text("Steller Intelligence App | Created for University Project")
