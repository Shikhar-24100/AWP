"""
Combined Streamlit App for Hugging Face Spaces
No separate API needed - everything runs in one process
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import pickle
from tensorflow import keras

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Antenna Pattern Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    """Load model and scaler"""
    try:
        model = keras.models.load_model('antenna_pattern_predictor.keras')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, False

MODEL, SCALER, MODEL_LOADED = load_model()

# ========== CLEAN PROFESSIONAL CSS ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #ffffff !important;
    }
    
    .main {
        background-color: #ffffff !important;
    }
    
    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
        background-color: #ffffff !important;
    }
    
    /* Header */
    .app-header {
        padding: 2.5rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 3rem;
    }
    
    .app-header h1 {
        color: #111827;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .app-header p {
        color: #6b7280;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.875rem;
        font-weight: 700;
        color: #111827;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        font-weight: 500;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Section headers */
    .section-header {
        color: #111827;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f9fafb;
        border-right: 1px solid #e5e7eb;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #111827;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] label {
        color: #374151;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.625rem 1.25rem;
        font-weight: 600;
        font-size: 0.9375rem;
        transition: background-color 0.2s;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #1d4ed8;
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background-color: #ffffff;
        color: #374151;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        padding: 0.625rem 1.25rem;
        font-weight: 500;
        transition: all 0.2s;
        width: 100%;
    }
    
    .stDownloadButton > button:hover {
        background-color: #f9fafb;
        border-color: #9ca3af;
    }
    
    /* Input fields */
    .stSelectbox label,
    .stNumberInput label,
    .stMultiSelect label {
        color: #374151;
        font-weight: 500;
        font-size: 0.875rem;
    }
    
    input, select {
        border-color: #d1d5db !important;
        border-radius: 6px !important;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 6px;
        border-left-width: 4px;
    }
    
    /* Feature cards */
    .feature-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-card h3 {
        color: #111827;
        font-size: 1.125rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
    }
    
    .feature-card p {
        color: #6b7280;
        font-size: 0.9375rem;
        margin: 0;
        line-height: 1.5;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f9fafb;
        border-radius: 6px;
        font-weight: 500;
        color: #374151;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Dividers */
    hr {
        margin: 2rem 0;
        border-color: #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# ========== PREDICTION LOGIC ==========

def encode_antenna_type(antenna_type: str):
    """One-hot encode antenna type"""
    type_dipole = 1 if antenna_type == 'dipole' else 0
    type_loop = 1 if antenna_type == 'loop' else 0
    type_monopole = 1 if antenna_type == 'monopole' else 0
    type_patch = 1 if antenna_type == 'patch' else 0
    return type_dipole, type_loop, type_monopole, type_patch

def predict_pattern(antenna_type, frequency, length, radius, width):
    """Make prediction using loaded model"""
    if MODEL is None or SCALER is None:
        return None
    
    try:
        # Check how many features the scaler expects
        n_features = SCALER.n_features_in_
        
        if n_features == 3:
            # Dipole-only model (frequency, length, radius)
            input_data = np.array([[frequency, length, radius]])
        elif n_features == 4:
            # Model with 4 features (frequency, length, radius, width)
            input_data = np.array([[frequency, length, radius, width]])
        elif n_features == 8:
            # Full model with antenna type encoding
            type_dipole, type_loop, type_monopole, type_patch = encode_antenna_type(antenna_type)
            input_data = np.array([[
                frequency, length, radius, width,
                type_dipole, type_loop, type_monopole, type_patch
            ]])
        else:
            st.error(f"Unexpected number of features: {n_features}")
            return None
        
        # Scale and predict
        input_scaled = SCALER.transform(input_data)
        prediction = MODEL.predict(input_scaled, verbose=0)[0]
        
        # Return result dictionary
        return {
            'gain_pattern': prediction.tolist(),
            'angles': list(range(360)),
            'max_gain_dbi': float(np.max(prediction)),
            'min_gain_dbi': float(np.min(prediction)),
            'avg_gain_dbi': float(np.mean(prediction)),
            'antenna_type': antenna_type,
            'frequency_ghz': frequency
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ========== VISUALIZATION FUNCTIONS ==========

def create_polar_plot(angles, gain, title="Radiation Pattern"):
    """Create polar plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=gain,
        theta=angles,
        mode='lines',
        name='Gain Pattern',
        line=dict(color='#2563eb', width=2.5),
        fill='toself',
        fillcolor='rgba(37, 99, 235, 0.1)',
        hovertemplate='Angle: %{theta}°<br>Gain: %{r:.2f} dBi<extra></extra>'
    ))
    
    max_idx = np.argmax(gain)
    fig.add_trace(go.Scatterpolar(
        r=[gain[max_idx]],
        theta=[angles[max_idx]],
        mode='markers',
        name='Peak',
        marker=dict(size=10, color='#dc2626'),
        hovertemplate=f'Peak: {angles[max_idx]}° at {gain[max_idx]:.2f} dBi<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#111827', family='Inter')),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[min(gain) - 1, max(gain) + 1],
                gridcolor='#d1d5db',
                tickfont=dict(color='#111827', size=11)
            ),
            angularaxis=dict(
                direction="clockwise",
                gridcolor='#d1d5db',
                tickfont=dict(color='#111827', size=11)
            ),
            bgcolor='#ffffff'
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(color='#111827', size=12)
        ),
        height=600,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(color='#111827', family='Inter')
    )
    
    return fig

def create_cartesian_plot(angles, gain, title="Radiation Pattern"):
    """Create cartesian plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=angles,
        y=gain,
        mode='lines',
        name='Gain',
        line=dict(color='#2563eb', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(37, 99, 235, 0.1)',
        hovertemplate='Angle: %{x}°<br>Gain: %{y:.2f} dBi<extra></extra>'
    ))
    
    max_idx = np.argmax(gain)
    min_idx = np.argmin(gain)
    
    fig.add_trace(go.Scatter(
        x=[angles[max_idx]],
        y=[gain[max_idx]],
        mode='markers',
        name='Peak',
        marker=dict(size=10, color='#10b981'),
        hovertemplate=f'Peak: {angles[max_idx]}° at {gain[max_idx]:.2f} dBi<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[angles[min_idx]],
        y=[gain[min_idx]],
        mode='markers',
        name='Null',
        marker=dict(size=10, color='#dc2626'),
        hovertemplate=f'Null: {angles[min_idx]}° at {gain[min_idx]:.2f} dBi<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#111827', family='Inter')),
        xaxis=dict(
            title=dict(text="Angle (degrees)", font=dict(color='#111827', size=13)),
            showgrid=True,
            gridcolor='#e5e7eb',
            color='#111827',
            tickfont=dict(color='#111827', size=11)
        ),
        yaxis=dict(
            title=dict(text="Gain (dBi)", font=dict(color='#111827', size=13)),
            showgrid=True,
            gridcolor='#e5e7eb',
            color='#111827',
            tickfont=dict(color='#111827', size=11)
        ),
        hovermode='x unified',
        height=500,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(color='#111827', family='Inter'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(color='#111827', size=12)
        )
    )
    
    return fig

def create_heatmap(angles, gain, title="Gain Heatmap"):
    """Create circular heatmap"""
    fig = go.Figure()
    
    angles_full = list(angles) + [0]
    gain_full = list(gain) + [gain[0]]
    
    colorscale = [
        [0, '#dc2626'],
        [0.5, '#fbbf24'],
        [1, '#10b981']
    ]
    
    fig.add_trace(go.Scatterpolar(
        r=gain_full,
        theta=angles_full,
        mode='markers',
        marker=dict(
            size=6,
            color=gain_full,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title="Gain (dBi)",
                thickness=15,
                len=0.7,
                tickfont=dict(color='#111827', size=11)
            )
        ),
        hovertemplate='Angle: %{theta}°<br>Gain: %{r:.2f} dBi<extra></extra>'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=gain_full,
        theta=angles_full,
        mode='lines',
        line=dict(color='rgba(37, 99, 235, 0.3)', width=1),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#111827', family='Inter')),
        polar=dict(
            angularaxis=dict(
                direction="clockwise",
                gridcolor='#d1d5db',
                tickfont=dict(color='#111827', size=11)
            ),
            radialaxis=dict(
                visible=True,
                range=[min(gain)-1, max(gain)+1],
                gridcolor='#d1d5db',
                tickfont=dict(color='#111827', size=11)
            ),
            bgcolor='#ffffff'
        ),
        height=600,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(color='#111827', family='Inter'),
        showlegend=False
    )
    
    return fig

def create_3d_pattern(angles, gain, title="3D Radiation Pattern"):
    """Create 3D pattern"""
    theta = np.radians(angles)
    phi = np.linspace(0, 2*np.pi, 50)
    
    THETA, PHI = np.meshgrid(theta, phi)
    R = np.tile(gain, (len(phi), 1))
    R = R - np.min(R) + 0.1
    
    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)
    
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=R,
        colorscale='Viridis',
        colorbar=dict(
            title="Gain (dBi)",
            thickness=15,
            len=0.7,
            tickfont=dict(color='#111827', size=11)
        ),
        hovertemplate='Gain: %{surfacecolor:.2f} dBi<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#111827', family='Inter')),
        scene=dict(
            xaxis=dict(
                showbackground=True,
                backgroundcolor='#ffffff',
                showgrid=True,
                gridcolor='#d1d5db',
                tickfont=dict(color='#111827', size=10)
            ),
            yaxis=dict(
                showbackground=True,
                backgroundcolor='#ffffff',
                showgrid=True,
                gridcolor='#d1d5db',
                tickfont=dict(color='#111827', size=10)
            ),
            zaxis=dict(
                showbackground=True,
                backgroundcolor='#ffffff',
                showgrid=True,
                gridcolor='#d1d5db',
                tickfont=dict(color='#111827', size=10)
            ),
            bgcolor='#ffffff'
        ),
        height=700,
        paper_bgcolor='#ffffff',
        font=dict(color='#111827', family='Inter')
    )
    
    return fig

# ========== MAIN APP ==========

def main():
    # Header
    st.markdown("""
    <div class="app-header">
        <h1>📡 Antenna Radiation Pattern Predictor</h1>
        <p>Deep learning model with MAE: 0.208 dB | R²: 0.9957</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not MODEL_LOADED:
        st.error("⚠️ Model files not found. Please ensure antenna_pattern_predictor.keras and scaler.pkl are uploaded.")
        st.stop()
    
    # Sidebar
    st.sidebar.header("Configuration")
    st.sidebar.markdown("---")
    
    # Example configurations
    examples = {
        "WiFi 2.4 GHz Dipole": ("dipole", 2.4, 0.5, 0.002, 0.0),
        "GPS L1 Patch": ("patch", 1.575, 0.49, 0.0, 0.49),
        "FM Radio Monopole": ("monopole", 0.1, 0.25, 0.001, 0.0),
        "Loop Antenna": ("loop", 1.0, 1.0, 0.01, 0.0)
    }
    
    example_names = ["Custom"] + list(examples.keys())
    selected_example = st.sidebar.selectbox("Quick Start", example_names)
    
    if selected_example != "Custom":
        default_type, default_freq, default_length, default_radius, default_width = examples[selected_example]
    else:
        default_type, default_freq, default_length, default_radius, default_width = "dipole", 2.4, 0.5, 0.002, 0.0
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Antenna Parameters")
    
    antenna_type = st.sidebar.selectbox(
        "Type",
        ["dipole", "monopole", "loop", "patch"],
        index=["dipole", "monopole", "loop", "patch"].index(default_type)
    )
    
    frequency = st.sidebar.number_input(
        "Frequency (GHz)",
        min_value=0.01,
        max_value=100.0,
        value=default_freq,
        step=0.1,
        format="%.2f"
    )
    
    length = st.sidebar.number_input(
        "Length (wavelengths)",
        min_value=0.01,
        max_value=5.0,
        value=default_length,
        step=0.01,
        format="%.3f"
    )
    
    radius = st.sidebar.number_input(
        "Radius (wavelengths)",
        min_value=0.0,
        max_value=1.0,
        value=default_radius,
        step=0.001,
        format="%.4f"
    )
    
    width = st.sidebar.number_input(
        "Width (wavelengths)",
        min_value=0.0,
        max_value=5.0,
        value=default_width,
        step=0.01,
        format="%.3f"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Visualization")
    
    viz_options = st.sidebar.multiselect(
        "Display Mode",
        ["Polar Plot", "Cartesian Plot", "Heatmap", "3D Surface"],
        default=["Polar Plot", "Cartesian Plot"]
    )
    
    st.sidebar.markdown("---")
    predict_btn = st.sidebar.button("Generate Pattern", type="primary")
    
    # Main content
    if predict_btn:
        with st.spinner("Generating radiation pattern..."):
            result = predict_pattern(antenna_type, frequency, length, radius, width)
        
        if result:
            st.markdown('<div class="section-header">Pattern Statistics</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Peak Gain", f"{result['max_gain_dbi']:.2f} dBi")
            with col2:
                st.metric("Null Depth", f"{result['min_gain_dbi']:.2f} dBi")
            with col3:
                st.metric("Average Gain", f"{result['avg_gain_dbi']:.2f} dBi")
            with col4:
                gain_range = result['max_gain_dbi'] - result['min_gain_dbi']
                st.metric("Gain Range", f"{gain_range:.2f} dB")
            
            st.markdown("---")
            
            angles = result['angles']
            gain = result['gain_pattern']
            
            if "Polar Plot" in viz_options:
                st.markdown('<div class="section-header">Polar Radiation Pattern</div>', unsafe_allow_html=True)
                fig_polar = create_polar_plot(angles, gain, f"{antenna_type.title()} at {frequency} GHz")
                st.plotly_chart(fig_polar, use_container_width=True)
            
            if "Cartesian Plot" in viz_options:
                st.markdown('<div class="section-header">Cartesian Analysis</div>', unsafe_allow_html=True)
                fig_cartesian = create_cartesian_plot(angles, gain, f"{antenna_type.title()} Gain Distribution")
                st.plotly_chart(fig_cartesian, use_container_width=True)
            
            if "Heatmap" in viz_options:
                st.markdown('<div class="section-header">Gain Heatmap</div>', unsafe_allow_html=True)
                fig_heatmap = create_heatmap(angles, gain, f"{antenna_type.title()} Heatmap")
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            if "3D Surface" in viz_options:
                st.markdown('<div class="section-header">3D Visualization</div>', unsafe_allow_html=True)
                fig_3d = create_3d_pattern(angles, gain, f"{antenna_type.title()} 3D Pattern")
                st.plotly_chart(fig_3d, use_container_width=True)
            
            with st.expander("Raw Pattern Data"):
                df = pd.DataFrame({
                    "Angle (°)": angles,
                    "Gain (dBi)": [f"{g:.3f}" for g in gain]
                })
                st.dataframe(df, use_container_width=True, height=400)
            
            st.markdown("---")
            st.markdown('<div class="section-header">Export Results</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = pd.DataFrame({
                    "Angle": angles,
                    "Gain_dBi": gain
                }).to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"{antenna_type}_{frequency}GHz.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_data = json.dumps(result, indent=2)
                
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"{antenna_type}_{frequency}GHz.json",
                    mime="application/json"
                )
    
    else:
        st.markdown('<div class="section-header">Getting Started</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3>Fast Predictions</h3>
                <p>Generate radiation patterns in under 1 second using our trained deep learning model.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3>Multiple Visualizations</h3>
                <p>View patterns in polar, cartesian, heatmap, or interactive 3D formats.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>High Accuracy</h3>
                <p>Model achieves MAE of 0.208 dB and R² of 0.9957 on test data.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3>Export Options</h3>
                <p>Download results in CSV or JSON format for further analysis.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("👈 Configure antenna parameters in the sidebar and click 'Generate Pattern' to begin.")

if __name__ == "__main__":
    main()
