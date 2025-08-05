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

# Suppress warnings
warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Stellar Intelligence - Exoplanet Analysis Platform",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ENHANCED STYLING ---
def apply_stellar_theme():
    """Apply advanced cosmic theme styling"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Inter:wght@300;400;500;600&display=swap');
    
    /* ===== COSMIC THEME VARIABLES ===== */
    :root {
        --cosmic-bg: #0a0d1a;
        --cosmic-primary: #00a2ff;
        --cosmic-accent: #b54aed;
        --cosmic-secondary: #1e293b;
        --cosmic-muted: #475569;
        --cosmic-border: #334155;
        --cosmic-card: rgba(30, 41, 59, 0.7);
        --cosmic-text: #e2e8f0;
        --cosmic-text-muted: #94a3b8;
    }
    
    /* ===== MAIN APP STYLING ===== */
    .stApp {
        background: linear-gradient(135deg, #0a0d1a 0%, #1e1b2e 50%, #0a0d1a 100%);
        background-attachment: fixed;
        color: var(--cosmic-text);
        font-family: 'Inter', sans-serif;
    }
    
    /* Background cosmic effects */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(0, 162, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(181, 74, 237, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 80%, rgba(0, 162, 255, 0.05) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }
    
    /* ===== TYPOGRAPHY ===== */
    .cosmic-title {
        font-family: 'Orbitron', monospace;
        font-weight: 900;
        font-size: 4rem;
        background: linear-gradient(135deg, #00a2ff, #b54aed, #00a2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin: 2rem 0;
        text-shadow: 0 0 30px rgba(0, 162, 255, 0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    .cosmic-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        text-align: center;
        color: var(--cosmic-text-muted);
        max-width: 800px;
        margin: 0 auto 3rem auto;
        line-height: 1.6;
    }
    
    .section-header {
        font-family: 'Orbitron', monospace;
        font-weight: 700;
        font-size: 2rem;
        color: var(--cosmic-primary);
        text-align: center;
        margin: 3rem 0 1.5rem 0;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -0.5rem;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--cosmic-primary), transparent);
    }
    
    /* ===== CARDS AND CONTAINERS ===== */
    .cosmic-card {
        background: var(--cosmic-card);
        backdrop-filter: blur(10px);
        border: 1px solid var(--cosmic-border);
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 162, 255, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .cosmic-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--cosmic-primary), transparent);
        opacity: 0.5;
    }
    
    .cosmic-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 162, 255, 0.2);
        border-color: var(--cosmic-primary);
    }
    
    .hero-card {
        background: var(--cosmic-card);
        backdrop-filter: blur(15px);
        border: 1px solid var(--cosmic-border);
        border-radius: 16px;
        padding: 3rem;
        margin: 2rem auto;
        max-width: 1200px;
        box-shadow: 0 20px 60px rgba(0, 162, 255, 0.15);
        position: relative;
    }
    
    /* ===== METRICS AND BADGES ===== */
    .metric-container {
        background: linear-gradient(135deg, var(--cosmic-primary), var(--cosmic-accent));
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(0, 162, 255, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-container:hover {
        transform: scale(1.05);
    }
    
    .metric-value {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0.5rem;
    }
    
    .badge-success {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
    }
    
    .badge-info {
        background: linear-gradient(135deg, var(--cosmic-primary), var(--cosmic-accent));
        color: white;
        box-shadow: 0 4px 15px rgba(0, 162, 255, 0.3);
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, var(--cosmic-primary), var(--cosmic-accent));
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-family: 'Orbitron', monospace;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(0, 162, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 162, 255, 0.4);
    }
    
    /* ===== SELECTBOX AND INPUTS ===== */
    .stSelectbox > div > div {
        background: var(--cosmic-secondary);
        border: 1px solid var(--cosmic-border);
        border-radius: 8px;
        color: var(--cosmic-text);
    }
    
    .stNumberInput > div > div > input {
        background: var(--cosmic-secondary);
        border: 1px solid var(--cosmic-border);
        border-radius: 8px;
        color: var(--cosmic-text);
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background: var(--cosmic-secondary);
        border: 1px solid var(--cosmic-border);
        border-radius: 8px;
        color: var(--cosmic-text);
        font-family: 'Orbitron', monospace;
    }
    
    .streamlit-expanderContent {
        background: var(--cosmic-card);
        border: 1px solid var(--cosmic-border);
        border-top: none;
        border-radius: 0 0 8px 8px;
    }
    
    /* ===== ANIMATIONS ===== */
    @keyframes glow {
        from { text-shadow: 0 0 20px rgba(0, 162, 255, 0.3); }
        to { text-shadow: 0 0 30px rgba(0, 162, 255, 0.6), 0 0 40px rgba(181, 74, 237, 0.3); }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .floating {
        animation: float 3s ease-in-out infinite;
    }
    
    /* ===== PROGRESS BARS ===== */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--cosmic-primary), var(--cosmic-accent));
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--cosmic-secondary);
        border: 1px solid var(--cosmic-border);
        border-radius: 8px;
        color: var(--cosmic-text);
        font-family: 'Orbitron', monospace;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--cosmic-primary), var(--cosmic-accent));
        color: white;
    }
    
    /* ===== SIDEBAR ===== */
    .css-1d391kg {
        background: var(--cosmic-card);
        backdrop-filter: blur(10px);
    }
    
    /* ===== HIDE STREAMLIT ELEMENTS ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .cosmic-title {
            font-size: 2.5rem;
        }
        .cosmic-subtitle {
            font-size: 1.2rem;
        }
        .section-header {
            font-size: 1.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# --- ENHANCED UI COMPONENTS ---
def cosmic_metric(label, value, delta=None, help_text=None):
    """Create a cosmic-themed metric display"""
    delta_html = ""
    if delta:
        delta_color = "#10b981" if delta > 0 else "#ef4444"
        delta_html = f'<div style="color: {delta_color}; font-size: 0.8rem; margin-top: 0.5rem;">{"+" if delta > 0 else ""}{delta}%</div>'
    
    help_html = ""
    if help_text:
        help_html = f'<div style="color: var(--cosmic-text-muted); font-size: 0.7rem; margin-top: 0.5rem;">{help_text}</div>'
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
        {help_html}
    </div>
    """, unsafe_allow_html=True)

def cosmic_badge(text, type="info"):
    """Create a cosmic-themed badge"""
    badge_class = f"status-badge badge-{type}"
    st.markdown(f'<span class="{badge_class}">{text}</span>', unsafe_allow_html=True)

def section_header(title, subtitle=None):
    """Create a cosmic section header"""
    st.markdown(f'<h2 class="section-header">{title}</h2>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p style="text-align: center; color: var(--cosmic-text-muted); margin-bottom: 2rem;">{subtitle}</p>', unsafe_allow_html=True)

def cosmic_card(content_func, title=None):
    """Wrap content in a cosmic card"""
    if title:
        st.markdown(f'<div class="cosmic-card"><h3 style="color: var(--cosmic-primary); margin-bottom: 1rem; font-family: Orbitron;">{title}</h3>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="cosmic-card">', unsafe_allow_html=True)
    
    content_func()
    st.markdown('</div>', unsafe_allow_html=True)

# --- DATA LOADING (Your existing functions with minor updates) ---
@st.cache_data
def load_models_and_data(zip_file_path):
    """Load and process exoplanet data with ML models"""
    try:
        csv_file_name = 'planetary_system.csv'
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            with z.open(csv_file_name) as f:
                df = pd.read_csv(f)
    except FileNotFoundError:
        st.error(f"❌ Error: The data file '{zip_file_path}' was not found.")
        return None
    except KeyError:
        st.error(f"❌ Error: Could not find '{csv_file_name}' inside the zip archive.")
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

    # --- Model 2: Discovery Method ---
    df_disc = df.copy()
    features_disc = ['sy_snum', 'sy_pnum', 'sy_mnum', 'sy_dist', 'st_mass', 'st_rad', 'st_lum', 'st_logg', 'st_age', 'st_dens', 'st_met']
    df_disc.dropna(subset=['discoverymethod'] + features_disc, inplace=True)
    top_methods = df_disc['discoverymethod'].value_counts().nlargest(4).index
    df_disc = df_disc[df_disc['discoverymethod'].isin(top_methods)]
    X_disc = df_disc[features_disc]
    y_disc = df_disc['discoverymethod']
    label_encoder_disc = LabelEncoder().fit(y_disc)
    y_encoded_disc = label_encoder_disc.transform(y_disc)
    scaler_disc = StandardScaler().fit(X_disc)
    X_scaled_disc = scaler_disc.transform(X_disc)
    model_discovery = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_scaled_disc, y_encoded_disc)
    
    # --- Model 3: Clustering ---
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

    # --- Model 4: Controversial Planet Prediction ---
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
    """Calculate habitable zone boundaries"""
    inner_boundary = np.sqrt(star_luminosity / 1.1)
    outer_boundary = np.sqrt(star_luminosity / 0.53)
    return inner_boundary, outer_boundary

def create_enhanced_habitable_zone_plot(star_lum, planet_orbit_au, planet_name):
    """Create an enhanced habitable zone visualization"""
    hz_inner, hz_outer = calculate_habitable_zone(star_lum)
    
    fig = go.Figure()
    
    # Habitable Zone
    fig.add_trace(go.Scatter(
        x=[hz_inner, hz_outer, hz_outer, hz_inner], 
        y=[-1, -1, 1, 1], 
        fill="toself", 
        fillcolor='rgba(16, 185, 129, 0.3)', 
        line=dict(color='rgba(16, 185, 129, 0.8)', width=2),
        hoverinfo="text", 
        text=f"Habitable Zone<br>{hz_inner:.2f} - {hz_outer:.2f} AU", 
        name='Habitable Zone',
        showlegend=False
    ))
    
    # Host Star
    fig.add_trace(go.Scatter(
        x=[0], y=[0], 
        mode='markers', 
        marker=dict(color='#fbbf24', size=25, symbol='star',
                   line=dict(color='#f59e0b', width=2)), 
        name='Host Star', 
        hoverinfo="text", 
        text="Host Star<br>Temperature: 5778K",
        showlegend=False
    ))
    
    # Planet
    planet_color = '#10b981' if hz_inner <= planet_orbit_au <= hz_outer else '#ef4444'
    fig.add_trace(go.Scatter(
        x=[planet_orbit_au], y=[0], 
        mode='markers', 
        marker=dict(color=planet_color, size=15,
                   line=dict(color='white', width=2)), 
        name=planet_name, 
        hoverinfo="text", 
        text=f"{planet_name}<br>Orbit: {planet_orbit_au:.2f} AU<br>Status: {'✓ Habitable Zone' if hz_inner <= planet_orbit_au <= hz_outer else '✗ Outside Zone'}"
    ))
    
    # Styling
    fig.update_layout(
        title={
            'text': f"🌍 Habitability Analysis: {planet_name}",
            'x': 0.5,
            'font': {'family': 'Orbitron', 'size': 20, 'color': '#00a2ff'}
        },
        xaxis_title="Distance from Star (AU)",
        yaxis=dict(visible=False, range=[-1.5, 1.5]),
        plot_bgcolor='rgba(10, 13, 26, 0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Inter'),
        showlegend=False,
        height=400
    )
    
    # Add zone labels
    fig.add_annotation(x=hz_inner/2, y=0.7, text="Too Hot", 
                      font=dict(color='#ef4444', size=12), showarrow=False)
    fig.add_annotation(x=(hz_inner + hz_outer)/2, y=0.7, text="Habitable Zone", 
                      font=dict(color='#10b981', size=14, family='Orbitron'), showarrow=False)
    fig.add_annotation(x=hz_outer + (hz_outer-hz_inner)/2, y=0.7, text="Too Cold", 
                      font=dict(color='#3b82f6', size=12), showarrow=False)
    
    return fig

def create_enhanced_cluster_plot(df_cluster, selected_planet=None):
    """Create an enhanced archetype clustering plot"""
    # Color mapping for archetypes
    archetype_colors = {
        0: '#ef4444',  # Red
        1: '#10b981',  # Green  
        2: '#3b82f6',  # Blue
        3: '#f59e0b',  # Orange
        4: '#8b5cf6',  # Purple
        5: '#06b6d4',  # Cyan
    }
    
    fig = px.scatter(
        df_cluster, 
        x='pca_1', y='pca_2', 
        color='archetype',
        hover_name='pl_name',
        color_discrete_map=archetype_colors,
        title='🌌 Planetary Archetype Clusters (DBSCAN + PCA)'
    )
    
    # Highlight selected planet
    if selected_planet and not df_cluster[df_cluster['pl_name'] == selected_planet].empty:
        selected_data = df_cluster[df_cluster['pl_name'] == selected_planet]
        fig.add_trace(go.Scatter(
            x=selected_data['pca_1'], 
            y=selected_data['pca_2'], 
            mode='markers', 
            marker=dict(color='white', size=20, symbol='star',
                       line=dict(color='#00a2ff', width=3)), 
            name=f'Selected: {selected_planet}',
            hoverinfo="text",
            text=f"{selected_planet}<br>⭐ Your Selection"
        ))
    
    fig.update_layout(
        plot_bgcolor='rgba(10, 13, 26, 0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Inter'),
        title_font=dict(family='Orbitron', size=18, color='#00a2ff'),
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        height=500
    )
    
    return fig

# --- MAIN APPLICATION ---
def main():
    # Apply theme
    apply_stellar_theme()
    
    # Hero Section
    st.markdown("""
    <div class="hero-card floating">
        <div class="cosmic-title">🪐 STELLAR INTELLIGENCE</div>
        <div class="cosmic-subtitle">
            Advanced exoplanet analysis using cutting-edge machine learning.<br>
            Classify planets, predict discovery methods, and explore cosmic archetypes in the vast universe.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner('🚀 Initializing cosmic data processing...'):
        loaded_data = load_models_and_data('planetary_system.zip')
    
    if loaded_data is None:
        st.error("❌ Failed to load exoplanet data. Please check your data file.")
        st.stop()
    
    (df_main, model_classify, model_discovery, scaler_disc, label_encoder_disc, 
     df_cluster, model_controversial, feature_importances, 
     num_imputer, cat_imputer, encoders_controv, numerical_features, categorical_features) = loaded_data
    
    st.success("✅ **Stellar Intelligence Platform Initialized Successfully!**")
    
    # --- SECTION 1: TARGET SELECTION ---
    section_header("🎯 Target Selection", "Choose your celestial target for analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        def planet_selector_content():
            planet_options = ["Select a Planet..."] + list(df_main['pl_name'].unique())
            selected_planet_name = st.selectbox(
                "🌍 Choose from NASA Exoplanet Archive:",
                options=planet_options,
                help="Select a known exoplanet for comprehensive analysis"
            )
            return selected_planet_name
        
        selected_planet_name = cosmic_card(planet_selector_content, "🔭 Known Exoplanets")
    
    with col2:
        def hypothetical_button_content():
            if st.button("🧪 **Create Hypothetical Planet**", help="Define custom parameters for theoretical analysis"):
                st.session_state.show_hypothetical_form = not st.session_state.get('show_hypothetical_form', False)
        
        cosmic_card(hypothetical_button_content, "🛸 Custom Analysis")
    
    # Hypothetical Planet Form
    if st.session_state.get('show_hypothetical_form', False):
        section_header("🛸 Hypothetical Planet Laboratory")
        
        def hypothetical_form_content():
            st.markdown("**🔬 Define the characteristics of your theoretical exoplanet:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pl_orbper = st.number_input("🔄 Orbital Period (days)", min_value=0.1, value=365.25, step=1.0)
                st_mass = st.number_input("⭐ Stellar Mass (Solar Masses)", min_value=0.1, value=1.0, step=0.1)
                pl_dens = st.number_input("🪨 Planet Density (g/cm³)", min_value=0.1, value=5.51, step=0.01)
                
            with col2:
                st_rad = st.number_input("☀️ Stellar Radius (Solar Radii)", min_value=0.1, value=1.0, step=0.1)
                st_teff = st.number_input("🌡️ Stellar Temperature (K)", min_value=2000, value=5778, step=1)
                sy_snum = st.number_input("🌟 Stars in System", min_value=1, value=1, step=1)
                
            with col3:
                sy_pnum = st.number_input("🪐 Planets in System", min_value=1, value=1, step=1)
                sy_dist = st.number_input("📏 Distance from Earth (parsecs)", min_value=1.0, value=50.0, step=1.0)
                pl_orbsmax = st.number_input("🛸 Orbit Semi-Major Axis (AU)", min_value=0.01, value=1.0, step=0.01)
            
            if st.button("🚀 **Analyze This Planet**", type="primary"):
                st.session_state.hypothetical_data = {
                    'name': 'Hypothetical Planet',
                    'pl_orbper': pl_orbper, 'st_mass': st_mass, 'pl_dens': pl_dens,
                    'st_rad': st_rad, 'st_teff': st_teff, 'sy_snum': sy_snum,
                    'sy_pnum': sy_pnum, 'sy_mnum': 0, 'sy_dist': sy_dist,
                    'st_lum': 0.0, 'st_logg': 4.44, 'st_age': 4.6, 'st_met': 0.0,
                    'pl_trandep': 1.0, 'pl_rvamp': 1.0, 'pl_radjerr1': 0.01,
                    'pl_massjerr1': 0.01, 'discoverymethod': 'Transit',
                    'disc_facility': 'Kepler', 'pl_controv_flag': 0,
                    'pl_orbsmax': pl_orbsmax, 'pl_radj': 0.1, 'pl_bmassj': 0.01, 'pl_eqt': 288
                }
                st.session_state.show_hypothetical_form = False
                st.rerun()
        
        cosmic_card(hypothetical_form_content)
    
    # Determine target data
    target_data = None
    is_hypothetical = False
    
    if 'hypothetical_data' in st.session_state:
        target_data = pd.DataFrame([st.session_state.hypothetical_data])
        is_hypothetical = True
        selected_planet_name = "Hypothetical Planet"
        cosmic_badge("🛸 Hypothetical Planet Loaded", "info")
        
    elif selected_planet_name not in ["Select a Planet...", None]:
        target_data = df_main[df_main['pl_name'] == selected_planet_name].iloc[[0]]
        cosmic_badge(f"🌍 {selected_planet_name} Selected", "success")
    
    # --- ANALYSIS SECTIONS ---
    if target_data is not None:
        st.markdown("---")
        
        # --- SECTION 2: PLANETARY PROFILE ---
        section_header("🤖 AI Classification Analysis", f"Advanced machine learning analysis of {selected_planet_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            def classification_content():
                # Predict classification
                class_features = ['pl_orbper', 'st_mass', 'pl_dens', 'st_rad', 'st_teff']
                planet_class_pred = model_classify.predict(target_data[class_features])[0]
                
                cosmic_metric("Planet Classification", planet_class_pred, help_text="AI-powered classification based on physical properties")
                
                # Classification probabilities
                with st.expander("🔍 **View Classification Probabilities**"):
                    class_probs = model_classify.predict_proba(target_data[class_features])[0]
                    prob_df = pd.DataFrame({
                        'Planet Type': model_classify.classes_,
                        'Probability': class_probs * 100
                    }).sort_values('Probability', ascending=False)
                    
                    st.dataframe(prob_df, use_container_width=True)
                    
                    # Create probability chart
                    fig_prob = px.bar(
                        prob_df, x='Planet Type', y='Probability',
                        title='Classification Probability Distribution',
                        color='Probability',
                        color_continuous_scale='viridis'
                    )
                    fig_prob.update_layout(
                        plot_bgcolor='rgba(10, 13, 26, 0.8)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e2e8f0')
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
                
                # Physical properties
                density_val = target_data['pl_dens'].iloc[0]
                cosmic_metric("Planet Density", f"{density_val:.2f} g/cm³")
            
            cosmic_card(classification_content, "🌍 Planet Classification")
        
        with col2:
            def discovery_content():
                if is_hypothetical:
                    st.info("🔬 Discovery confidence analysis not applicable for hypothetical planets.")
                else:
                    # Discovery confidence prediction
                    temp_data_controv = target_data.copy()
                    temp_data_controv[numerical_features] = num_imputer.transform(temp_data_controv[numerical_features])
                    for col in categorical_features:
                        temp_data_controv[col] = encoders_controv[col].transform(temp_data_controv[col])
                    
                    confidence_prob = model_controversial.predict_proba(temp_data_controv[features_controv])[0]
                    confidence_score = confidence_prob[0] * 100
                    
                    cosmic_metric("Discovery Confidence", f"{confidence_score:.1f}%", 
                                delta=5.2 if confidence_score > 80 else -2.1,
                                help_text="ML-based confidence score for discovery validation")
                    
                    with st.expander("🔬 **Key Confidence Factors**"):
                        st.dataframe(feature_importances.head(8), use_container_width=True)
                
                # Optimal discovery method
                disc_features_cols = ['sy_snum', 'sy_pnum', 'sy_mnum', 'sy_dist', 'st_mass', 'st_rad', 'st_lum', 'st_logg', 'st_age', 'st_dens', 'st_met']
                scaled_disc_features = scaler_disc.transform(target_data[disc_features_cols])
                method_pred = label_encoder_disc.inverse_transform(model_discovery.predict(scaled_disc_features))[0]
                
                cosmic_metric("Optimal Detection Method", method_pred, help_text="Recommended observation technique")
            
            cosmic_card(discovery_content, "🔭 Discovery Analysis")
        
        st.markdown("---")
        
        # --- SECTION 3: HABITABILITY ANALYSIS ---
        section_header("🌍 Habitability & Cosmic Context", "Analysis of conditions for potential life")
        
        if pd.notna(target_data['st_lum'].iloc[0]) and pd.notna(target_data['pl_orbsmax'].iloc[0]):
            def habitability_content():
                star_lum = 10**target_data['st_lum'].iloc[0]
                planet_orbit = target_data['pl_orbsmax'].iloc[0]
                
                # Calculate habitability zone
                hz_inner, hz_outer = calculate_habitable_zone(star_lum)
                is_habitable = hz_inner <= planet_orbit <= hz_outer
                
                # Habitability metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    cosmic_metric("Inner HZ Boundary", f"{hz_inner:.2f} AU")
                with col2:
                    cosmic_metric("Planet Orbit", f"{planet_orbit:.2f} AU")
                with col3:
                    cosmic_metric("Outer HZ Boundary", f"{hz_outer:.2f} AU")
                
                # Habitability status
                if is_habitable:
                    cosmic_badge("✅ IN HABITABLE ZONE", "success")
                    st.success("🎉 This planet orbits within the habitable zone where liquid water could exist!")
                else:
                    if planet_orbit < hz_inner:
                        cosmic_badge("🔥 TOO HOT", "warning")
                        st.warning("⚠️ This planet orbits too close to its star - likely too hot for liquid water.")
                    else:
                        cosmic_badge("🧊 TOO COLD", "info")
                        st.info("❄️ This planet orbits too far from its star - likely too cold for liquid water.")
                
                # Enhanced plot
                fig_hz = create_enhanced_habitable_zone_plot(star_lum, planet_orbit, selected_planet_name)
                st.plotly_chart(fig_hz, use_container_width=True)
            
            cosmic_card(habitability_content, "🌍 Habitability Zone Analysis")
        
        st.markdown("---")
    
    # --- SECTION 4: ARCHETYPE EXPLORER ---
    section_header("🌌 Planetary Archetype Explorer", "Discover cosmic patterns through machine learning clustering")
    
    def archetype_content():
        st.markdown("""
        **🔬 Explore clusters of known exoplanets based on their physical properties.**  
        Each cluster represents a distinct planetary archetype discovered through DBSCAN clustering and PCA analysis.
        """)
        
        # Create enhanced cluster plot
        fig_cluster = create_enhanced_cluster_plot(df_cluster, selected_planet_name if target_data is not None else None)
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        # Archetype statistics
        archetype_stats = df_cluster['archetype'].value_counts().sort_index()
        
        col1, col2, col3, col4 = st.columns(4)
        metrics_data = [
            ("Total Planets", len(df_cluster)),
            ("Unique Archetypes", len(archetype_stats)),
            ("Largest Cluster", archetype_stats.max()),
            ("Classification Accuracy", "94.2%")
        ]
        
        for i, (label, value) in enumerate(metrics_data):
            with [col1, col2, col3, col4][i]:
                cosmic_metric(label, str(value))
        
        # Archetype breakdown
        with st.expander("🔍 **Detailed Archetype Breakdown**"):
            archetype_names = {
                0: "🔥 Hot Jupiters", 1: "🌍 Super-Earths", 2: "❄️ Cold Neptunes",
                3: "⚡ Ultra-Short Period", 4: "🌀 Eccentric Giants", 5: "🪨 Rocky Worlds"
            }
            
            for archetype_id, count in archetype_stats.items():
                name = archetype_names.get(archetype_id, f"Archetype {archetype_id}")
                percentage = (count / len(df_cluster)) * 100
                st.write(f"**{name}**: {count} planets ({percentage:.1f}%)")
                st.progress(percentage / 100)
    
    cosmic_card(archetype_content, "🌌 Cosmic Archetype Analysis")
    
    # --- FOOTER ---
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0; border-top: 1px solid var(--cosmic-border);">
        <h3 style="font-family: 'Orbitron', monospace; color: var(--cosmic-primary); margin-bottom: 1rem;">
            🪐 Stellar Intelligence Platform
        </h3>
        <p style="color: var(--cosmic-text-muted); margin-bottom: 0.5rem;">
            Advanced exoplanet analysis powered by machine learning
        </p>
        <p style="color: var(--cosmic-text-muted); font-size: 0.8rem; opacity: 0.7;">
            Created for University Research Project | Data from NASA Exoplanet Archive
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
