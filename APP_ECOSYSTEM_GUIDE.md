# 🚀 ML Alloy Project - Complete App Ecosystem

**Three integrated web applications for alloy discovery and machine learning**

---

## 📱 The Three Apps

Your ML Alloy Project now has **three powerful Streamlit applications** that work together:

| # | App Name | Port | Purpose | Launcher |
|---|----------|------|---------|----------|
| **1** | **Materials Project Explorer** | 8501 | Download DFT data, train ML models, discover alloys | `run_webapp.bat` |
| **2** | **Experimental Data Manager** | 8502 | Upload and visualize temperature-dependent properties | `run_experimental_app.bat` |
| **3** | **CALPHAD Phase Diagrams** | 8503 | Calculate phase diagrams and thermodynamic properties | `run_calphad_app.bat` |

**You can run all three simultaneously!** Each operates independently on its own port.

---

## 🎯 Quick Start Guide

### First Time Setup

1. **Activate Environment**
   ```
   Double-click: START_PROJECT.bat
   ```

2. **Install CALPHAD Tools** (one-time)
   ```
   Double-click: install_calphad.bat
   ```

3. **Test Installation**
   ```bash
   python test_calphad.py
   ```

### Launch Applications

**Option 1: Launch All Three** (recommended for full workflow)
```
1. Double-click: run_webapp.bat
2. Double-click: run_experimental_app.bat
3. Double-click: run_calphad_app.bat
```

Your browser will have 3 tabs:
- http://localhost:8501 - Materials Project data
- http://localhost:8502 - Experimental curves
- http://localhost:8503 - Phase diagrams

**Option 2: Launch One at a Time**
- Choose the app you need
- Double-click its launcher
- Start working!

---

## 🔄 Complete Workflow: From Data to Discovery

### Workflow 1: ML-Driven Alloy Discovery

**Goal:** Find new high-strength alloys using machine learning

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Gather Computational Data (App 1 - Port 8501)      │
├─────────────────────────────────────────────────────────────┤
│ 1. Open Materials Project Explorer                          │
│ 2. Go to "Download Data" page                               │
│ 3. Search for alloy system (e.g., Fe-Ni, Ti-Al)            │
│ 4. Filter by properties (e.g., e_above_hull < 0.1)         │
│ 5. Download dataset as CSV                                  │
│ 6. Preview in "Explore Data" page                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Add Thermodynamics (App 3 - Port 8503)             │
├─────────────────────────────────────────────────────────────┤
│ 1. Open CALPHAD Phase Diagrams                              │
│ 2. Load TDB database for same system                        │
│ 3. Calculate phase diagram (0-100% composition)             │
│ 4. For each composition in CSV:                             │
│    - Calculate equilibrium at processing T (e.g., 1000K)   │
│    - Extract: stable phases, phase fractions, Gibbs energy │
│ 5. Save equilibrium results as JSON                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Add Experimental Data (App 2 - Port 8502)          │
├─────────────────────────────────────────────────────────────┤
│ 1. Open Experimental Data Manager                           │
│ 2. Upload strength vs temperature curves                    │
│ 3. Upload hardness vs temperature curves                    │
│ 4. Compare materials from same alloy family                 │
│ 5. Export combined dataset                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Combine and Train ML Model (App 1 - Port 8501)     │
├─────────────────────────────────────────────────────────────┤
│ 1. Merge all datasets:                                      │
│    - MP data (0K DFT properties)                            │
│    - CALPHAD data (finite-T thermodynamics)                 │
│    - Experimental data (real-world performance)             │
│ 2. Go to "Train Models" page                                │
│ 3. Upload combined CSV                                       │
│ 4. Select features and target property                      │
│ 5. Train model (Random Forest, XGBoost, Neural Net)        │
│ 6. Evaluate performance (R², RMSE, MAE)                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Discover New Alloys (App 1 - Port 8501)            │
├─────────────────────────────────────────────────────────────┤
│ 1. Go to "Discover Alloys" page                             │
│ 2. Define composition space to explore                      │
│ 3. Use trained model to predict properties                  │
│ 4. Filter by desired criteria (e.g., strength > 1000 MPa)  │
│ 5. Export top candidates for experimental validation        │
└─────────────────────────────────────────────────────────────┘
```

**Result:** List of promising new alloy compositions, prioritized by ML predictions!

---

### Workflow 2: Validate CALPHAD Predictions

**Goal:** Check if thermodynamic database accurately predicts experimental behavior

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Calculate Phase Diagram (App 3 - Port 8503)        │
├─────────────────────────────────────────────────────────────┤
│ 1. Load TDB database (e.g., Ti-Al)                          │
│ 2. Calculate binary phase diagram                           │
│ 3. Export high-resolution figure                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Overlay Experimental Data (App 2 - Port 8502)      │
├─────────────────────────────────────────────────────────────┤
│ 1. Upload DSC data (phase transition temperatures)          │
│ 2. Upload XRD data (phase identification)                   │
│ 3. Plot experimental points on phase diagram image          │
│ 4. Compare predicted vs measured transition temperatures    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Assess Accuracy                                     │
├─────────────────────────────────────────────────────────────┤
│ 1. Calculate deviation: |T_predicted - T_measured|          │
│ 2. If good agreement (<20K): Database is accurate ✓        │
│ 3. If poor agreement (>50K): Database needs refinement ✗   │
│ 4. Use ESPEI to optimize parameters (Phase 4 of plan)      │
└─────────────────────────────────────────────────────────────┘
```

---

### Workflow 3: Alloy Design for Specific Application

**Goal:** Design alloy for high-temperature turbine blade

**Requirements:**
- Single-phase FCC at 1200K (avoid phase transformations)
- Formation energy < 0 (stable)
- Density < 8 g/cm³ (lightweight)

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Survey Composition Space (App 1 - Port 8501)       │
├─────────────────────────────────────────────────────────────┤
│ 1. Download Ni-Cr-Al ternary alloys from Materials Project │
│ 2. Filter: e_above_hull < 0.05, density < 8                │
│ 3. Get ~50 candidates                                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Check Phase Stability (App 3 - Port 8503)          │
├─────────────────────────────────────────────────────────────┤
│ 1. Load Ni-Cr-Al TDB database                               │
│ 2. For each candidate composition:                          │
│    - Calculate equilibrium at 1200K                         │
│    - Check if single-phase FCC                              │
│    - If yes: Keep ✓                                         │
│    - If no (multi-phase): Discard ✗                        │
│ 3. Narrow to ~20 candidates                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Validate with Experiments (App 2 - Port 8502)      │
├─────────────────────────────────────────────────────────────┤
│ 1. For similar alloys (Inconel, Waspaloy):                  │
│    - Upload high-temp strength curves                       │
│    - Upload creep data                                       │
│ 2. Identify composition-property trends                     │
│ 3. Predict performance of 20 candidates                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Final Ranking (Manual or App 1 ML)                 │
├─────────────────────────────────────────────────────────────┤
│ 1. Rank by predicted strength at 1200K                      │
│ 2. Select top 3-5 for experimental synthesis                │
│ 3. Export composition recipes                               │
└─────────────────────────────────────────────────────────────┘
```

**Result:** Optimized alloy compositions ready for lab testing!

---

## 📊 App 1: Materials Project Explorer (Port 8501)

### What It Does
- Downloads computational (DFT) data from Materials Project
- Explores 150,000+ materials database
- Trains machine learning models
- Predicts properties of new alloys

### Key Features
- **Download Data**: Search and download by elements, properties, crystal structure
- **Explore Data**: Visualize properties, filter datasets, generate statistics
- **Train Models**: Random Forest, XGBoost, Neural Networks with cross-validation
- **Discover Alloys**: Use trained models to find promising new compositions

### When to Use
- Starting a new project (download initial dataset)
- Need 0K DFT properties (formation energy, band gap, bulk modulus)
- Training ML models
- Screening large composition spaces

### Documentation
- `README_WEBAPP.md` - Detailed user guide
- `BUG_FIX_SUMMARY.md` - Known issues and fixes

---

## 📈 App 2: Experimental Data Manager (Port 8502)

### What It Does
- Manages experimental temperature-dependent properties
- Visualizes strength, hardness, conductivity vs temperature
- Compares multiple materials
- Integrates with CALPHAD predictions

### Key Features
- **Upload Data**: Import CSV files with T vs property curves
- **Visualize Curves**: Interactive plots with smoothing and derivatives
- **Compare Materials**: Overlay multiple datasets for comparison
- **Manage Data**: Browse, delete, export experimental datasets

### When to Use
- Have experimental data to upload (DSC, tensile tests, hardness)
- Need to compare temperature-dependent behavior
- Validating computational predictions
- Extracting features for ML (e.g., peak strength temperature)

### Documentation
- `EXPERIMENTAL_DATA_GUIDE.md` - Complete user manual

---

## 🔥 App 3: CALPHAD Phase Diagrams (Port 8503)

### What It Does
- Calculates thermodynamic equilibria using CALPHAD method
- Generates binary and ternary phase diagrams
- Predicts phase stability at any temperature and composition
- Simulates solidification (Scheil method)

### Key Features
- **Load Database**: Upload or select TDB thermodynamic databases
- **Binary Phase Diagram**: Calculate and visualize T vs composition diagrams
- **Equilibrium Calculator**: Get stable phases and fractions at specific conditions
- **Scheil Simulation**: Model non-equilibrium solidification (coming soon)

### When to Use
- Need finite-temperature thermodynamics (>0K)
- Designing heat treatments
- Predicting phase transformations
- Understanding solidification behavior
- Validating experimental phase observations

### Documentation
- `CALPHAD_APP_GUIDE.md` - Complete user manual
- `CALPHAD_SETUP_GUIDE.md` - Installation and background
- `CALPHAD_QUICK_START.md` - 10-minute tutorial

---

## 🧪 Example: Complete Analysis of Ti-6Al-4V

Let's analyze the most common titanium alloy using all three apps!

### Background
**Ti-6Al-4V**: 90% Ti, 6% Al, 4% V (by weight)
- Most widely used titanium alloy
- Aerospace, medical, automotive applications
- Two-phase (α + β) microstructure

---

### Part 1: Computational Data (App 1 - Port 8501)

**Action:**
1. Launch `run_webapp.bat`
2. Go to "Download Data"
3. Search elements: Ti, Al, V
4. Download all Ti-Al-V ternary compounds
5. Save as `ti_al_v_materials.csv`

**Result:**
- 25 stable Ti-Al-V phases from Materials Project
- For each: formation energy, volume, density
- Ti-6Al-4V composition should have low e_above_hull (stable)

---

### Part 2: Phase Diagram (App 3 - Port 8503)

**Action:**
1. Launch `run_calphad_app.bat`
2. Upload Ti-Al-V TDB database (download from NIMS)
3. Calculate **Ti-Al binary** diagram (500-2000K)
4. Calculate **Ti-V binary** diagram (500-2000K)
5. Calculate equilibrium at 900°C (1173K) for 90Ti-6Al-4V composition

**Result:**
- Ti-Al diagram shows α (HCP) and β (BCC) regions
- At 1173K, Ti-6Al-4V: ~60% α + 40% β (predicted)
- Matches known two-phase microstructure ✓

**Screenshot:**
```
Phase Fractions at 1173K, Ti-6Al-4V:
  HCP_A3 (α):  62.3%
  BCC_A2 (β):  37.7%
```

---

### Part 3: Experimental Validation (App 2 - Port 8502)

**Action:**
1. Launch `run_experimental_app.bat`
2. Upload Ti-6Al-4V tensile strength vs temperature (find in literature)
3. Upload Ti-6Al-4V hardness vs temperature
4. Compare with pure Ti and Ti-6Al-2Sn-4Zr-2Mo (Ti-6242)

**Data format (CSV):**
```csv
Temperature_C,Yield_Strength_MPa
20,880
200,760
400,620
600,380
800,180
```

**Result:**
- Ti-6Al-4V maintains strength up to 400°C
- Above 600°C, rapid strength loss
- Better than pure Ti, similar to Ti-6242

---

### Part 4: ML Prediction (App 1 - Port 8501)

**Action:**
1. Combine datasets:
   ```python
   import pandas as pd

   # MP data
   mp_data = pd.read_csv('ti_al_v_materials.csv')

   # CALPHAD data (manually created)
   calphad_data = pd.DataFrame({
       'composition': ['Ti6Al4V', ...],
       'alpha_fraction_900C': [0.62, ...],
       'beta_fraction_900C': [0.38, ...]
   })

   # Experimental data
   exp_data = pd.DataFrame({
       'composition': ['Ti6Al4V'],
       'yield_strength_RT': [880],
       'yield_strength_400C': [620]
   })

   # Merge
   combined = mp_data.merge(calphad_data).merge(exp_data)
   combined.to_csv('ti_alloys_combined.csv')
   ```

2. Go to "Train Models" in App 1
3. Upload `ti_alloys_combined.csv`
4. Train to predict yield_strength_RT from:
   - formation_energy (MP)
   - alpha_fraction_900C (CALPHAD)
   - density (MP)

**Result:**
- R² = 0.85 (good prediction)
- Key feature: α fraction (more α = higher strength)
- Can now predict strength of new Ti-Al-V compositions!

---

### Part 5: Design New Alloy

**Goal:** Design Ti-Al-V alloy with higher strength than Ti-6Al-4V

**Strategy:**
1. From CALPHAD: Higher Al content → more α phase
2. From MP: Need to stay stable (e_above_hull < 0.1)
3. From ML: Predict strength for Ti-8Al-4V, Ti-10Al-4V, etc.

**Action:**
1. In App 3: Calculate equilibrium for Ti-8Al-4V at 1173K
   - Result: 75% α + 25% β (more α ✓)

2. In App 1: Check if Ti-8Al-4V exists in MP
   - Check formation energy
   - If stable, predict strength using ML model
   - Predicted yield strength: 920 MPa (vs 880 for Ti-6Al-4V)

3. In App 2: Compare processing window
   - Upload Ti-8Al-4V phase transition data (if available)
   - Check if processing temperature changes

**Result:**
- **Ti-8Al-4V** predicted to have 5% higher strength
- Still processable at similar temperatures
- Candidate for experimental synthesis!

---

## 🎓 Learning Path

### Beginner (Week 1)
1. ✅ Activate environment and test installation
2. ✅ Launch App 1, download a simple binary system (Fe-Ni)
3. ✅ Explore data in "Explore Data" page
4. ✅ Launch App 3, load sample database, calculate Al-Zn diagram
5. ✅ Upload example experimental data to App 2

### Intermediate (Week 2-3)
1. Download your research alloy system from Materials Project
2. Find TDB database for same system (NIMS)
3. Calculate phase diagram and equilibria
4. Combine MP + CALPHAD data
5. Train first ML model

### Advanced (Week 4+)
1. Integrate experimental data from your lab
2. Build custom TDB database with ESPEI
3. Automate workflow (Python scripts)
4. Deploy models for inverse design
5. Scheil solidification simulations

---

## 🔧 Technical Details

### System Requirements
- **OS**: Windows 10/11 (or Linux/Mac with modifications)
- **Python**: 3.9+ (tested with 3.13.9)
- **RAM**: 8 GB minimum, 16 GB recommended
- **Disk**: 5 GB for packages, 50+ GB for databases

### Installed Packages

**Core:**
- `mp-api` - Materials Project access
- `pymatgen` - Materials science toolkit
- `pycalphad` - CALPHAD calculations
- `streamlit` - Web application framework

**ML & Data:**
- `scikit-learn` - Machine learning
- `xgboost` - Gradient boosting
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `plotly` - Visualization

**Optional:**
- `espei` - CALPHAD parameter optimization
- `scheil` - Solidification simulation
- `symengine` - Fast symbolic math

### Port Configuration

| Port | App | Configurable In |
|------|-----|----------------|
| 8501 | Materials Project | `run_webapp.bat` |
| 8502 | Experimental Data | `run_experimental_app.bat` |
| 8503 | CALPHAD | `run_calphad_app.bat` |

To change ports, edit the batch file:
```batch
streamlit run app.py --server.port 8501
```

### Data Storage

```
ML-Alloy-Project/
├── downloaded_data/          # App 1 - MP datasets (CSV)
├── trained_models/           # App 1 - Saved ML models
├── example_data/             # App 2 - Experimental curves
├── uploaded_experimental/    # App 2 - User uploads
├── calphad_databases/        # App 3 - TDB files
└── calphad_outputs/          # App 3 - Phase diagrams, equilibria
```

---

## 🐛 Troubleshooting

### Problem: Port already in use

**Error:** `Port 8501 is already in use`

**Solution:**
1. Close other Streamlit apps
2. Or change port in batch file
3. Or kill process:
   ```bash
   netstat -ano | findstr :8501
   taskkill /PID <number> /F
   ```

---

### Problem: App won't start

**Error:** Module not found, import errors

**Solution:**
1. Make sure environment is activated
2. Reinstall packages:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements_calphad.txt
   ```

---

### Problem: Database loading fails

**Error:** Cannot load TDB file in App 3

**Solution:**
1. Check file format (must be .tdb or .TDB)
2. Verify TDB syntax (use test_calphad.py)
3. Try sample database (alzn_sample.tdb) first
4. Check PyCalphad version (need 0.11.0+)

---

## 📚 Documentation Index

### Getting Started
- `START_HERE.txt` - First steps
- `ACTIVATION_GUIDE.md` - Environment activation
- `HOW_TO_ACTIVATE.md` - Detailed activation instructions

### Main Application (App 1)
- `README_WEBAPP.md` - Complete guide
- `BUG_FIX_SUMMARY.md` - Known issues

### Experimental Data (App 2)
- `EXPERIMENTAL_DATA_GUIDE.md` - User manual

### CALPHAD (App 3)
- `CALPHAD_APP_GUIDE.md` - User manual (this file)
- `CALPHAD_SETUP_GUIDE.md` - Installation and concepts
- `CALPHAD_QUICK_START.md` - 10-minute tutorial

### Overall Project
- `PROJECT_PLAN.md` - Development roadmap
- `API_SETUP_GUIDE.md` - Materials Project API
- `APP_ECOSYSTEM_GUIDE.md` - Multi-app integration (YOU ARE HERE)

---

## 🎯 Next Development Phases

Based on the original CALPHAD integration plan, we've completed:

✅ **Phase 1**: Installation & Setup
- PyCalphad installed
- Sample databases provided
- Testing framework created

✅ **Phase 2**: Basic Streamlit Integration
- Interactive phase diagram calculator
- Equilibrium calculator
- Database management

🔄 **Phase 3** (Next): Materials Project + CALPHAD Integration
- Automatic feature extraction from CALPHAD
- Batch equilibrium calculations
- Combined dataset creation

📋 **Phase 4** (Future): Experimental Data Integration
- Overlay CALPHAD on experimental curves
- ESPEI parameter optimization
- Database refinement workflow

📋 **Phase 5** (Future): Advanced Features
- Ternary phase diagrams
- Property diagrams (T-dependent Cp, H, S)
- Scheil solidification interface

📋 **Phase 6** (Future): Full ML Integration
- Auto-generate features for any composition
- Inverse design: "Find alloy with X strength at Y temperature"
- Uncertainty quantification

---

## 🚀 Quick Reference

### Daily Workflow

**To start working:**
```
1. Double-click: START_PROJECT.bat
2. Double-click: run_webapp.bat (and/or other apps)
3. Browser opens automatically
```

**To download new data:**
```
1. App 1, "Download Data" page
2. Enter elements or formula
3. Set filters
4. Download CSV
```

**To calculate phase diagram:**
```
1. App 3, "Load Database" page
2. Select/upload TDB file
3. "Binary Phase Diagram" page
4. Set elements and T range
5. Calculate
```

**To upload experimental data:**
```
1. App 2, "Upload Data" page
2. Choose CSV file (T, Property columns)
3. Enter metadata
4. Upload and save
```

**To train ML model:**
```
1. Prepare combined CSV (MP + CALPHAD + Experimental)
2. App 1, "Train Models" page
3. Upload CSV
4. Select features and target
5. Train and evaluate
```

---

## ✅ Quick Checklist

Before starting a new project, ensure:

- [ ] Environment activated (`START_PROJECT.bat`)
- [ ] PyCalphad installed and tested (`test_calphad.py` passes)
- [ ] Materials Project API key set (already done: b6qrDfF03...)
- [ ] At least one TDB database available
- [ ] All three apps can launch successfully
- [ ] Have target alloy system in mind
- [ ] Know desired properties to optimize

---

## 🎉 You're All Set!

You now have a complete ecosystem for:
- ✅ Downloading computational data (DFT)
- ✅ Calculating thermodynamics (CALPHAD)
- ✅ Managing experimental data
- ✅ Training ML models
- ✅ Discovering new alloys

**Happy discovering! 🔬⚗️🔥**

---

*ML Alloy Discovery Project*
*Digital Foundry*
*Three Apps, One Goal: Accelerate Materials Innovation*
