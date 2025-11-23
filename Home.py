"""
Digital Foundry - Materials Science Toolkit
Multi-page Streamlit Application

Main landing page with navigation to all three apps:
1. Materials Project Explorer
2. GNN Property Predictor
3. CALPHAD Thermodynamic Tools
"""

import streamlit as st
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Digital Foundry - Materials Science Toolkit",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .app-card {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .app-card-2 {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .app-card-3 {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">🔥 Digital Foundry</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Materials Science Toolkit</div>', unsafe_allow_html=True)

st.markdown("---")

# Introduction
st.markdown("""
## Welcome to the Digital Foundry Materials Science Toolkit!

This integrated platform provides powerful tools for materials discovery and analysis:
""")

# Two columns for currently available apps
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📊 Materials Project Explorer

    **Discover and analyze materials data**

    - Query Materials Project database
    - Search by elements, properties
    - Download datasets as CSV
    - Visualize material properties
    - Train ML models (sklearn)
    - Export for ML training

    **Use Case:** Find candidate materials and build predictive models
    """)
    if st.button("🚀 Launch Materials Project Explorer", key="app1", use_container_width=True):
        st.switch_page("pages/1_Materials_Project_Explorer.py")

with col2:
    st.markdown("""
    ### 🔥 CALPHAD Tools

    **Thermodynamic modeling and analysis**

    - Phase diagram calculation
    - Equilibrium predictions
    - Create TDB from DFT data (ESPEI)
    - Temperature-property curves
    - Scheil solidification

    **Use Case:** Calculate phase diagrams and thermodynamic properties
    """)
    if st.button("🚀 Launch CALPHAD Tools", key="app3", use_container_width=True):
        st.switch_page("pages/3_CALPHAD_Tools.py")

st.markdown("---")

# Workflow section
st.markdown("""
## 🔄 Integrated Workflow

These tools work together for comprehensive materials design:

```
1. Materials Project Explorer
   ↓ Download materials data → CSV
   ↓ Train ML models for property predictions

2. CALPHAD Tools (ESPEI)
   ↓ Generate TDB files from MP data
   ↓ Calculate phase diagrams
   ↓ Add thermodynamic features to datasets

Result: Complete materials characterization with ML + Thermodynamics!
```

**Coming Soon:** Advanced GNN-based property predictors
""")

# Feature highlights
st.markdown("---")
st.markdown("## ✨ Key Features")

feature_col1, feature_col2 = st.columns(2)

with feature_col1:
    st.markdown("""
    **Data Access & Discovery:**
    - 🌐 Materials Project API integration
    - 📈 Advanced filtering and search
    - 💾 CSV export functionality
    - 📊 Interactive visualizations

    **Machine Learning:**
    - 🤖 Sklearn-based ML models
    - 🎯 Property prediction (formation energy, stability, density)
    - 📉 Training and evaluation metrics
    - 📈 Model comparison (Linear, Random Forest, Gradient Boosting)
    """)

with feature_col2:
    st.markdown("""
    **Thermodynamic Modeling:**
    - 🌡️ Binary phase diagram calculation
    - ⚖️ Equilibrium predictions
    - 🔬 ESPEI integration (DFT → TDB)
    - 📊 Multi-phase support (LIQUID, FCC, BCC, HCP)

    **Integration:**
    - 🔄 Seamless data flow between apps
    - 📁 File-based data exchange
    - 💾 Export CALPHAD features for ML training
    - 📚 Comprehensive documentation
    """)

# Quick start
st.markdown("---")
st.markdown("## 🚀 Quick Start")

tab1, tab2, tab3 = st.tabs(["New to Materials Science?", "Researcher", "Developer"])

with tab1:
    st.markdown("""
    **Perfect for learning!**

    1. Start with **Materials Project Explorer** to browse existing materials
    2. Download sample datasets to understand material properties
    3. Train simple ML models on the downloaded data
    4. Explore **CALPHAD Tools** to calculate phase diagrams
    5. Generate TDB files from DFT data using ESPEI

    **Tip:** Each app has built-in help and documentation!
    """)

with tab2:
    st.markdown("""
    **Accelerate your research!**

    **For materials discovery:**
    1. Query MP for candidate materials → App 1
    2. Train ML models to predict properties → App 1
    3. Calculate phase diagrams and thermodynamic properties → App 2

    **For CALPHAD database development:**
    1. Export MP data as CSV → App 1
    2. Generate multi-phase TDB files with ESPEI → App 2
    3. Calculate phase diagrams to validate → App 2

    **For ML feature engineering:**
    1. Collect training data → App 1
    2. Add CALPHAD features (phase fractions, Gibbs energy) → App 2
    3. Train improved ML models with combined features → App 1
    """)

with tab3:
    st.markdown("""
    **Extend and customize!**

    **Architecture:**
    - Streamlit multi-page app structure
    - Modular espei_integration module
    - CSV-based data exchange between apps

    **Key Libraries:**
    - Materials Project API (mp-api)
    - PyCalphad (thermodynamic calculations)
    - ESPEI (CALPHAD database generation)
    - Scikit-learn (ML models)
    - Plotly & Matplotlib (visualizations)

    **Customization Ideas:**
    - Add advanced ML features (matminer descriptors)
    - Implement ternary phase diagrams
    - Add ESPEI MCMC optimization
    - Create pre-computed TDB database
    - Build GNN models with PyTorch Geometric

    **Documentation:** See README.md and APP_ANALYSIS_COMPLETE.md
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Digital Foundry Materials Science Toolkit</strong></p>
    <p>Integrating Materials Project, Machine Learning, and CALPHAD Thermodynamics</p>
    <p>Built with Streamlit • PyCalphad • ESPEI • Scikit-learn</p>
    <p style="font-size: 0.9em; margin-top: 1rem;">
        <a href="https://github.com" style="color: #666; text-decoration: none;">📚 Documentation</a> •
        <a href="https://github.com" style="color: #666; text-decoration: none;">⭐ Star on GitHub</a>
    </p>
</div>
""", unsafe_allow_html=True)
