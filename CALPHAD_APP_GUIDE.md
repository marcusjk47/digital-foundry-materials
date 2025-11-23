# 🔥 CALPHAD Phase Diagram Calculator - User Guide

**Interactive web application for thermodynamic calculations and phase diagram visualization**

---

## 🚀 Quick Start

### Launch the App

Double-click to run:
```
run_calphad_app.bat
```

The app will open in your browser at: **http://localhost:8503**

---

## 📱 App Overview

The CALPHAD app has **5 main pages**:

### 1️⃣ **Home** - Introduction and Status
- Check PyCalphad installation status
- View available databases
- Quick links to resources

### 2️⃣ **Load Database** - Import TDB Files
- Select from existing databases
- Upload new TDB files
- Inspect database contents (elements, phases)

### 3️⃣ **Binary Phase Diagram** - Calculate Phase Diagrams
- Select 2-component system
- Set temperature and composition ranges
- Generate and visualize phase diagrams
- Export high-resolution figures

### 4️⃣ **Equilibrium Calculator** - Point Calculations
- Calculate equilibrium at specific T and composition
- Binary or ternary systems
- View stable phases and phase fractions
- Get thermodynamic properties (Gibbs energy, chemical potentials)

### 5️⃣ **Scheil Simulation** - Solidification Modeling
- (Coming soon) Non-equilibrium solidification
- Predict microsegregation
- Calculate solidification path

---

## 📖 Detailed Walkthrough

### Step 1: Load a Database

**Option A: Use Example Database**

1. Go to **"Load Database"** page
2. Select **"Select Existing Database"**
3. Choose `alzn_sample.tdb` (provided)
4. Click **"Load and Inspect Database"**
5. ✅ You should see:
   - Elements: AL, ZN
   - Phases: FCC_A1, HCP_A3, LIQUID

**Option B: Upload Your Own**

1. Go to **"Load Database"** page
2. Select **"Upload New Database"**
3. Click **"Choose a TDB file"**
4. Select your .tdb file
5. Preview the contents
6. Click **"Save and Load Database"**

**Where to Get TDB Files:**
- **NIMS Database**: https://cpddb.nims.go.jp/ (free, 700+ systems)
- **Research Papers**: Many include TDB files as supplementary data
- **Create Your Own**: Use ESPEI to optimize from DFT data

---

### Step 2: Calculate a Binary Phase Diagram

**Example: Al-Zn System**

1. **Load Database** (alzn_sample.tdb)

2. Go to **"Binary Phase Diagram"** page

3. **Select Elements:**
   - Element A: `AL`
   - Element B: `ZN`

4. **Select Phases:**
   - Check: `FCC_A1`, `HCP_A3`, `LIQUID`
   - (Or select all)

5. **Set Parameters:**
   - Min Temperature: `300 K`
   - Max Temperature: `1500 K`
   - Temperature Step: `10 K`
   - Composition Step: `0.01` (1%)
   - Pressure: `101325 Pa` (1 atm)

6. Click **"Calculate Phase Diagram"** 🔥

7. **Wait for calculation** (30 seconds to 2 minutes)

8. **View Results:**
   - Interactive phase diagram
   - Temperature vs Mole Fraction Zn
   - Phase boundaries clearly labeled

9. **Save Figure:**
   - Enter filename: `al_zn_phase_diagram.png`
   - Select DPI: `150` or `300`
   - Click **"Save Figure"**
   - ✅ Saved to: `calphad_outputs/`

---

### Step 3: Calculate Equilibrium at Specific Conditions

**Example: What phases are stable at 600K with 30% Zn?**

1. **Load Database** (alzn_sample.tdb)

2. Go to **"Equilibrium Calculator"** page

3. **Select System Type:**
   - Choose: `Binary (2 components)`

4. **Select Components:**
   - Component A: `AL`
   - Component B: `ZN`

5. **Select Phases:**
   - Check all available phases

6. **Set Conditions:**
   - Temperature: `600 K`
   - Pressure: `101325 Pa`
   - Mole Fraction ZN: `0.30` (30%)

7. Click **"Calculate Equilibrium"** ⚖️

8. **View Results:**
   - **Stable Phases**: FCC_A1, HCP_A3 (example)
   - **Phase Fractions**: Bar chart showing % of each phase
   - **Molar Gibbs Energy**: Total system energy
   - **Chemical Potentials**: For each component

9. **Export Results:**
   - Enter filename: `equilibrium_600K_30Zn.json`
   - Click **"Save Results"**
   - ✅ JSON file with all data saved

---

## 🎯 Practical Use Cases

### Use Case 1: Alloy Design

**Goal:** Find composition range where single FCC phase is stable at processing temperature

1. Calculate binary phase diagram (e.g., Al-Zn)
2. Identify single-phase FCC region
3. Select composition within that region
4. Use equilibrium calculator to verify at processing T (e.g., 500K)
5. Confirm only FCC_A1 is stable

**Result:** Alloy composition that won't form unwanted phases during heat treatment

---

### Use Case 2: Heat Treatment Planning

**Goal:** Determine safe annealing temperature for Al-30%Zn alloy

1. Load database and calculate phase diagram
2. Draw vertical line at X(Zn) = 0.30
3. Identify phase transitions (e.g., FCC ↔ FCC+HCP boundary)
4. Choose annealing temperature in single-phase region
5. Verify with equilibrium calculator

**Result:** Optimized heat treatment temperature that maintains desired microstructure

---

### Use Case 3: Compare with Experimental Data

**Goal:** Validate thermodynamic database against measured phase transitions

1. Calculate phase diagram for your system
2. Export high-resolution figure
3. Overlay experimental data points (e.g., from DSC, XRD)
4. Compare predicted vs measured transition temperatures
5. Assess database accuracy

**Result:** Confidence in using CALPHAD predictions for new compositions

---

### Use Case 4: Integration with Materials Project

**Goal:** Combine DFT (0K) with CALPHAD (finite T) for ML training

**Workflow:**
1. Download compositions from Materials Project (app.py)
2. For same system, calculate CALPHAD phase diagram (calphad_app.py)
3. For each MP composition, calculate equilibrium at processing T (e.g., 800K)
4. Extract features:
   - **From MP**: Formation energy, band gap, bulk modulus (0K)
   - **From CALPHAD**: Stable phases at T, phase fractions, Gibbs energy
5. Combine into ML dataset
6. Train model that predicts both 0K and finite-T properties

**Result:** ML model with temperature-dependent predictions

---

## 🔧 Advanced Features

### Multi-Port Architecture

The ML Alloy Project has **3 Streamlit apps** running on different ports:

| App | Port | Purpose | Launcher |
|-----|------|---------|----------|
| **Main App** | 8501 | Materials Project data, ML training | `run_webapp.bat` |
| **Experimental App** | 8502 | Temperature curves, experimental data | `run_experimental_app.bat` |
| **CALPHAD App** | 8503 | Phase diagrams, thermodynamics | `run_calphad_app.bat` |

**You can run all 3 simultaneously!**

Open 3 browser tabs:
- http://localhost:8501 - Download alloy data from Materials Project
- http://localhost:8502 - Upload experimental strength vs temperature
- http://localhost:8503 - Calculate phase diagram for same system

---

### Calculation Performance Tips

**For faster calculations:**

1. **Reduce temperature steps**
   - Use 10-20K steps instead of 5K for initial exploration
   - Refine later for publication-quality diagrams

2. **Reduce composition steps**
   - Start with 0.02 (2%) steps
   - Use 0.01 (1%) or 0.005 (0.5%) for final diagrams

3. **Limit phase selection**
   - Only include phases likely to be stable
   - Exclude phases known to be metastable in your T range

4. **Install symengine** (if not already installed)
   ```bash
   conda install -c conda-forge symengine
   ```
   - Speeds up symbolic math by 10-100x

**Typical calculation times:**
- Binary diagram (300-1500K, 0.01 composition step): 30-120 seconds
- Equilibrium point calculation: <1 second
- Ternary diagram: 5-30 minutes (coming soon)

---

## 📁 Output Files

All outputs are saved to: `calphad_outputs/`

**Generated files:**
- **Phase diagrams**: `*.png` (figures)
- **Equilibrium results**: `*.json` (structured data)
- **Calculation logs**: `*.txt` (metadata)

**Example output structure:**
```
calphad_outputs/
├── al_zn_phase_diagram.png
├── equilibrium_600K_30Zn.json
├── fe_cr_phase_diagram_300dpi.png
└── ti_al_equilibrium_results.json
```

---

## 🐛 Troubleshooting

### Problem: "PyCalphad is not installed"

**Solution:**
1. Run the installer: `install_calphad.bat`
2. Or manually install:
   ```bash
   conda install -c conda-forge pycalphad
   ```
3. Restart the app

---

### Problem: "No database loaded"

**Solution:**
1. Go to **"Load Database"** page
2. Either select existing or upload new TDB file
3. Click load/inspect button
4. Wait for ✅ success message

---

### Problem: Calculation is very slow

**Possible causes:**

1. **Too many temperature/composition points**
   - Reduce steps (see Performance Tips above)

2. **Symengine not installed**
   - Install: `conda install -c conda-forge symengine`

3. **Complex database with many phases**
   - Limit phase selection to relevant phases

---

### Problem: "Calculation failed" error

**Common causes:**

1. **Temperature out of range**
   - Check TDB validity range (usually at top of TDB file)
   - Most databases: 298-6000K

2. **Incompatible phases**
   - Some phases may not be defined for all elements
   - Try with fewer phases

3. **Numerical convergence issues**
   - Try different temperature range
   - Adjust composition range
   - Check TDB file for errors

---

### Problem: Phase diagram looks wrong

**Verification steps:**

1. **Compare to literature**
   - Search for published phase diagrams of same system
   - Check ASM Handbook, NIMS database

2. **Check database source**
   - Is it a recent assessment?
   - Has it been validated experimentally?

3. **Verify element selection**
   - Did you select the correct elements?
   - Check spelling (case-sensitive!)

4. **Check pressure**
   - Most databases are for 1 atm (101325 Pa)
   - Using different pressure may give unexpected results

---

## 🎓 Learning Resources

### Understanding Phase Diagrams

**Key concepts:**
- **Liquidus**: Temperature where last liquid solidifies (cooling) or first solid melts (heating)
- **Solidus**: Temperature where first liquid forms (heating) or last liquid solidifies (cooling)
- **Phase Field**: Region where specific phase(s) are stable
- **Tie Line**: Horizontal line connecting compositions of coexisting phases
- **Lever Rule**: Calculate phase fractions from tie line

**Reading a binary diagram:**
1. **X-axis**: Composition (mole or weight fraction of element B)
2. **Y-axis**: Temperature (usually K or °C)
3. **Regions**: Labeled with stable phase(s)
4. **Lines**: Phase boundaries (transitions)

**Example interpretation:**
- At 600K, 30% Zn: Read vertically to find which region
- If in "FCC_A1 + HCP_A3" region: Two phases coexist
- If in "FCC_A1" region: Single phase stable

---

### CALPHAD Method

**What is CALPHAD?**
- **CAL**culation of **PHAD**e **D**iagrams
- Uses thermodynamic models to predict phase equilibria
- Based on Gibbs energy minimization

**Key equation:**
At equilibrium, the system minimizes total Gibbs energy:
```
G_total = Σ(n_i * G_i)
```
where:
- n_i = moles of phase i
- G_i = Gibbs energy of phase i

**Model hierarchy:**
1. **Unary**: Pure element (GHSERAL, GHSERZN)
2. **Binary**: Two elements (interaction parameters)
3. **Ternary**: Three elements (higher-order interactions)
4. **Multicomponent**: 4+ elements

---

### PyCalphad Documentation

**Official resources:**
- **Docs**: https://pycalphad.org/docs/
- **Examples**: https://pycalphad.org/docs/latest/examples/
- **API Reference**: https://pycalphad.org/docs/latest/api/

**Recommended tutorials:**
1. "Getting Started with PyCalphad"
2. "Binary Phase Diagram Tutorial"
3. "Equilibrium Calculations"
4. "Property Diagrams"

---

## 🔗 Integration with Other Apps

### Workflow: Materials Project → CALPHAD → ML

**Step-by-step integration:**

1. **Download DFT Data** (Main App - Port 8501)
   - Go to "Download Data" page
   - Select system (e.g., Fe-Cr)
   - Download compositions
   - Save as CSV

2. **Calculate CALPHAD Properties** (CALPHAD App - Port 8503)
   - Load Fe-Cr TDB database
   - For each composition in CSV:
     - Calculate equilibrium at processing T (e.g., 1000K)
     - Extract stable phases, phase fractions
     - Save results

3. **Combine Datasets** (Python script)
   ```python
   import pandas as pd

   # Load MP data
   mp_data = pd.read_csv('fe_cr_alloys.csv')

   # Load CALPHAD results
   calphad_data = pd.read_json('calphad_outputs/equilibrium_results.json')

   # Merge on composition
   combined = mp_data.merge(calphad_data, on='composition')

   # Now have: formation_energy (0K), band_gap (0K),
   #           stable_phases (1000K), phase_fractions (1000K)
   ```

4. **Train ML Model** (Main App - Port 8501)
   - Go to "Train Models" page
   - Upload combined CSV
   - Select features (DFT + CALPHAD)
   - Train and evaluate

**Result:** ML model that predicts stability at both 0K and finite temperatures!

---

## 📊 Example: Complete Analysis

**Goal:** Design Al-Zn alloy for high strength at room temperature

### Phase 1: CALPHAD Analysis

1. **Load database**: `alzn_sample.tdb`

2. **Calculate phase diagram**:
   - Elements: AL, ZN
   - T range: 200-800K (covers room temp)
   - Composition: 0-100% Zn

3. **Identify regions**:
   - 0-5% Zn: Single-phase FCC at 300K ✓
   - 5-30% Zn: FCC + HCP at 300K (two-phase) ✓✓
   - 30-100% Zn: Mostly HCP at 300K

4. **Design choice**:
   - For single-phase: <5% Zn
   - For two-phase strengthening: 5-30% Zn
   - Decision: **Al-10%Zn** (two-phase for strength)

### Phase 2: Equilibrium Verification

1. **Calculate equilibrium** at 300K, 10% Zn:
   - Stable phases: FCC_A1 (80%), HCP_A3 (20%)
   - Confirms two-phase microstructure

2. **Check processing temperature** (500K, 10% Zn):
   - Stable phases: FCC_A1 (100%)
   - Can process as single phase, precipitates form on cooling

### Phase 3: Materials Project Comparison

1. **Search MP** for Al-Zn compositions near 10% Zn
2. **Compare**:
   - MP formation energy (0K DFT)
   - CALPHAD Gibbs energy (300K)
   - Experimental strength data

3. **Validate**:
   - Do CALPHAD predictions match MP stability trends?
   - Do predicted phases match experimental XRD?

### Phase 4: ML Prediction

1. **Combine data**:
   - MP: E_form, bulk_modulus (0K)
   - CALPHAD: phase_fractions (300K, 500K)
   - Experimental: yield_strength

2. **Train model**:
   - Input: Composition + phase fractions
   - Output: Yield strength

3. **Discover**:
   - Predict strength for untested compositions
   - Identify optimal Al-Zn ratio

---

## 🎉 Next Steps

**Now that you have the CALPHAD app running:**

✅ **Immediate tasks:**
1. Install PyCalphad: Run `install_calphad.bat`
2. Test installation: Run `test_calphad.py`
3. Launch app: Run `run_calphad_app.bat`
4. Calculate first phase diagram (Al-Zn example)

✅ **Short term:**
1. Download more TDB files from NIMS (Fe-Cr, Cu-Ni, Ti-Al)
2. Calculate phase diagrams for your alloy systems
3. Compare CALPHAD predictions with Materials Project data
4. Integrate phase diagram features into ML dataset

✅ **Long term (Phase 3-6 of plan):**
1. Automate CALPHAD feature extraction
2. Build custom databases with ESPEI
3. Scheil solidification simulations
4. Full integration: MP + CALPHAD + Experimental → ML

---

## 📞 Support

**Documentation:**
- Setup Guide: `CALPHAD_SETUP_GUIDE.md`
- Quick Start: `CALPHAD_QUICK_START.md`
- This Guide: `CALPHAD_APP_GUIDE.md`

**External Resources:**
- PyCalphad: https://pycalphad.org/
- NIMS Database: https://cpddb.nims.go.jp/
- CALPHAD Journal: https://www.sciencedirect.com/journal/calphad

**Testing:**
- Test script: `test_calphad.py`
- Example database: `calphad_databases/alzn_sample.tdb`

---

**🔬 Happy phase diagram calculating!**

*Part of the ML Alloy Discovery Project*
