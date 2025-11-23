# 💪 Temperature-Strength Curves Guide

**Predict mechanical property evolution with temperature using CALPHAD microstructure calculations**

---

## Overview

The **Temperature-Property Curves** feature in the CALPHAD app allows you to:

✅ Calculate how phase fractions change with temperature
✅ Predict strength/hardness based on microstructure
✅ Compare CALPHAD predictions with experimental data
✅ Generate temperature-strength curves for alloy design
✅ Export data for machine learning training

---

## How It Works

### The Connection: Microstructure → Properties

**Key Principle:** Mechanical properties depend on microstructure (which phases are present and in what amounts)

**Workflow:**
```
Temperature → CALPHAD → Phase Fractions → Property Model → Strength/Hardness
```

**Example:**
- At 300K: 90% FCC (strong) + 10% HCP (weak) → High strength
- At 800K: 30% FCC + 70% LIQUID (very weak) → Low strength

---

## Three Prediction Methods

### Method 1: Rule of Mixtures

**Best for:** Quick estimates when you know properties of pure phases

**Formula:**
```
Property = Σ (Phase_Fraction_i × Phase_Property_i)
```

**Example: Al-Zn Yield Strength**
- FCC_A1 strength = 200 MPa
- HCP_A3 strength = 150 MPa
- At 400K: 60% FCC + 40% HCP
- **Predicted strength = 0.6×200 + 0.4×150 = 180 MPa**

**Advantages:**
- Simple and intuitive
- No calibration needed
- Good for initial screening

**Limitations:**
- Ignores phase boundary strengthening
- Doesn't account for temperature-dependent phase properties
- Assumes linear mixing

---

### Method 2: Empirical Coefficients

**Best for:** When you have some experimental data to fit parameters

**Formula:**
```
Property = A + B×T + C×f_FCC + D×f_HCP + E×f_LIQUID + ...
```

Where:
- A = Baseline property
- B = Temperature dependence coefficient
- C, D, E = Phase contribution coefficients
- T = Temperature (K)
- f_i = Phase fractions

**Example:**
```
Yield_Strength = 1000 - 0.5×T + 200×f_FCC + 100×f_HCP - 500×f_LIQUID
```

At 600K with 70% FCC, 30% HCP:
```
= 1000 - 0.5×600 + 200×0.7 + 100×0.3
= 1000 - 300 + 140 + 30
= 870 MPa
```

**Advantages:**
- More flexible than Rule of Mixtures
- Can capture temperature effects explicitly
- Can fit to experimental data

**How to Get Coefficients:**
1. Measure strength at several temperatures
2. Calculate phase fractions at those temperatures (CALPHAD)
3. Fit linear regression: `Strength = f(T, phase_fractions)`
4. Use fitted coefficients in the app

---

### Method 3: Experimental Data Comparison

**Best for:** Validating CALPHAD predictions against real measurements

**What It Does:**
- You upload experimental temperature-strength data
- App calculates CALPHAD phase fractions at same temperatures
- Displays both side-by-side for correlation analysis

**Use Cases:**
- **Validate databases:** Does phase evolution match strength changes?
- **Identify mechanisms:** Which phase transformation causes strength drop?
- **Improve models:** Use correlation to build better empirical models

---

## Step-by-Step Tutorial

### Example: Al-30%Zn Yield Strength Prediction

#### Step 1: Launch App

```
run_calphad_app.bat
```

Navigate to: **"💪 Temperature-Property Curves"** page

---

#### Step 2: Load Database

If not already loaded:
1. Go to "📁 Load Database"
2. Select `alzn_sample.tdb`
3. Load and verify

---

#### Step 3: Define Composition

**Settings:**
- Base Element: `AL`
- Alloying Element: `ZN`
- Mole Fraction of ZN: `0.30` (30%)

**Result:** Composition is Al-30%Zn

---

#### Step 4: Select Phases

Keep all phases selected:
- ✅ FCC_A1
- ✅ HCP_A3
- ✅ LIQUID

---

#### Step 5: Set Temperature Range

**Settings:**
- Minimum Temperature: `300 K` (room temperature)
- Maximum Temperature: `800 K` (moderate high temperature)
- Temperature Step: `20 K` (for speed)
- Pressure: `101325 Pa` (1 atm)

**Result:** Will calculate at 26 temperature points

---

#### Step 6: Choose Property Model

**Option A: Rule of Mixtures** (simplest)

1. Select "Rule of Mixtures"
2. Property: "Yield Strength (MPa)"
3. Enter phase properties:
   - FCC_A1 value: `200 MPa` (aluminum is ~200 MPa at RT)
   - HCP_A3 value: `150 MPa` (zinc is ~150 MPa at RT)
   - LIQUID value: `10 MPa` (liquid metal has very low strength)

---

**Option B: Empirical Coefficients** (more realistic)

1. Select "Empirical Coefficients"
2. Coefficient A (baseline): `1000 MPa`
3. Coefficient B (temp dependence): `-0.5` (strength decreases with T)
4. FCC_A1 coefficient: `200`
5. HCP_A3 coefficient: `100`
6. LIQUID coefficient: `-800` (liquid greatly weakens alloy)

**This captures:**
- Base strength of 1000 MPa
- Loss of 0.5 MPa per K increase
- FCC phase contributes +200 MPa per unit fraction
- Liquid phase strongly reduces strength

---

**Option C: Experimental Comparison**

If you have experimental data:

1. Prepare CSV file:
   ```csv
   Temperature_K,Property_Value
   300,210
   400,180
   500,140
   600,90
   700,40
   800,15
   ```

2. Upload the file
3. App will show experimental curve + CALPHAD phases

---

#### Step 7: Calculate

Click **"🔥 Calculate Temperature-Property Curve"**

**Progress:**
- "Calculating T = 300 K (1/26)..."
- "Calculating T = 320 K (2/26)..."
- ...
- "Complete!"

**Time:** ~5-10 seconds for 26 points

---

#### Step 8: View Results

**Plot 1: Temperature vs Yield Strength**
- X-axis: Temperature (°C)
- Y-axis: Yield Strength (MPa)
- Blue line: CALPHAD prediction
- Shows how strength decreases with temperature

**Expected Behavior:**
- Room temp (27°C): High strength (~180-200 MPa)
- Mid temp (200-400°C): Gradual decrease
- High temp (500°C+): Rapid decrease as liquid forms
- Very high temp (527°C = 800K): Very low strength due to melting

**Plot 2: Phase Evolution**
- Shows how phase fractions change with temperature
- Helps explain why strength changes

**Example Interpretation:**
- At 27°C: 100% FCC_A1 → Maximum strength
- At 300°C: FCC_A1 starts transforming to HCP_A3 → Slight strength decrease
- At 450°C: LIQUID appears → Rapid strength drop
- At 527°C: Mostly LIQUID → Minimal strength

---

#### Step 9: Analyze Data Table

The results table shows for each temperature:
- Temperature_K
- Temperature_C
- Property_Value (predicted strength)
- FCC_A1_fraction
- HCP_A3_fraction
- LIQUID_fraction

**Use this to:**
- Identify critical temperatures (where phases change)
- Find maximum operating temperature (before strength drops too much)
- Design heat treatment cycles

---

#### Step 10: Export for ML

Click **"💾 Save Results as CSV"**

**Filename:** `Al30%Zn_temp_property.csv`

**Result:** CSV file in `calphad_outputs/` folder

**Use in ML Training:**
1. Open App 1 (Materials Project Explorer)
2. Go to "Train Models" page
3. Upload this CSV along with MP data
4. Now ML model has both:
   - 0K DFT properties (from Materials Project)
   - Finite-T phase fractions (from CALPHAD)
   - → Better predictions!

---

## Practical Applications

### Application 1: Determine Maximum Operating Temperature

**Problem:** Need to design Al-Zn alloy for high-temperature application

**Approach:**
1. Calculate temperature-strength curve
2. Define minimum acceptable strength (e.g., 100 MPa)
3. Find temperature where strength drops below threshold
4. Set maximum operating temperature slightly below that

**Example Result:**
- Al-30%Zn maintains >100 MPa up to 450°C
- **Maximum operating temperature: 420°C** (with safety margin)

---

### Application 2: Optimize Composition for Temperature Resistance

**Problem:** Which Al-Zn composition has best high-temp strength?

**Approach:**
1. Calculate curves for multiple compositions:
   - Al-10%Zn
   - Al-20%Zn
   - Al-30%Zn
   - Al-40%Zn

2. Plot all curves on same graph
3. Compare strength at target operating temperature (e.g., 400°C)

**Example Result:**
- Al-10%Zn: 150 MPa at 400°C ← Best
- Al-20%Zn: 120 MPa at 400°C
- Al-30%Zn: 90 MPa at 400°C
- Al-40%Zn: 60 MPa at 400°C

**Conclusion:** Lower Zn content gives better high-temp performance

---

### Application 3: Explain Experimental Observations

**Problem:** Measured yield strength drops sharply at 450°C. Why?

**Approach:**
1. Upload experimental data
2. Compare with CALPHAD phase evolution
3. Identify correlation

**Example Finding:**
- Experimental strength drop at 450°C
- CALPHAD shows LIQUID phase appears at 455°C
- **Conclusion:** Strength drop due to incipient melting
- **Action:** Modify composition to raise liquidus temperature

---

### Application 4: Build Empirical Models for ML

**Problem:** Need strength prediction model for ML training

**Approach:**
1. Collect experimental strength data at various temperatures
2. Calculate CALPHAD phase fractions at same temperatures
3. Fit empirical coefficients using regression
4. Use fitted model to predict strength for new compositions

**Workflow:**
```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load experimental data
exp = pd.read_csv('al_zn_strength_experimental.csv')

# Load CALPHAD phase fractions (from this app)
calphad = pd.read_csv('al_zn_phase_fractions.csv')

# Merge
data = exp.merge(calphad, on='Temperature_K')

# Prepare features and target
X = data[['Temperature_K', 'FCC_A1_fraction', 'HCP_A3_fraction', 'LIQUID_fraction']]
y = data['Yield_Strength_MPa']

# Fit model
model = LinearRegression()
model.fit(X, y)

# Get coefficients
print(f"Coefficient B (temp): {model.coef_[0]:.2f}")
print(f"Coefficient C (FCC): {model.coef_[1]:.2f}")
print(f"Coefficient D (HCP): {model.coef_[2]:.2f}")
print(f"Coefficient E (LIQUID): {model.coef_[3]:.2f}")

# Now use these coefficients in the app!
```

---

## Integration with Other Apps

### With App 1 (Materials Project)

**Export CALPHAD features for ML training:**

1. Download compositions from Materials Project
2. For each composition, calculate phase fractions vs temperature
3. Export as CSV
4. Merge with MP data
5. Train ML model with combined features

**Enhanced ML Features:**
- From MP: formation_energy, band_gap, bulk_modulus (0K)
- **From CALPHAD: FCC_fraction_800K, HCP_fraction_800K** (finite T)
- → Model learns both 0K stability and finite-T microstructure!

---

### With App 2 (Experimental Data Manager)

**Validate CALPHAD predictions:**

1. Upload experimental strength curves to App 2
2. Calculate CALPHAD phase evolution (App 3)
3. Compare in App 3 using "Upload Experimental Data" mode
4. Identify correlations and discrepancies

**Example Analysis:**
- Experimental: Strength drop at 450°C
- CALPHAD: LIQUID appears at 455°C
- Correlation: ✅ Good agreement (5K difference is acceptable)
- Conclusion: Database is accurate for this system

---

## Advanced Tips

### Tip 1: Phase-Specific Properties

For Rule of Mixtures, use literature values or estimates:

**Common phase strengths (room temperature approximations):**
- FCC (Al-rich): 50-300 MPa depending on purity
- HCP (Zn-rich): 100-200 MPa depending on grain size
- BCC (Fe-rich): 200-800 MPa depending on carbon content
- LIQUID: 1-10 MPa (negligible)
- Intermetallic compounds: 100-1000 MPa (often brittle)

**Temperature dependence:**
- Most metals lose ~0.5-2 MPa per K
- Liquid phase appears → strength drops dramatically
- Precipitate strengthening varies with phase fraction

---

### Tip 2: Validate with Known Alloys

Before predicting new compositions, validate model:

1. Choose well-studied alloy (e.g., Al-6061, Ti-6Al-4V)
2. Find experimental strength vs temperature data
3. Calculate with your model
4. Compare predictions vs experiments
5. Adjust phase properties or coefficients to match
6. **Now** apply to new compositions

---

### Tip 3: Account for Grain Size Effects

Rule of Mixtures assumes fixed grain size. For more accuracy:

**Hall-Petch strengthening:**
```
Strength = Strength_lattice + k / sqrt(grain_size)
```

**Modified approach:**
1. Use Rule of Mixtures for lattice strength
2. Add grain size term separately
3. Or include grain size as empirical coefficient

---

### Tip 4: Multi-Phase Contributions

For complex alloys with 3+ phases:

**Approach 1: Rule of Mixtures**
```
Strength = f_phase1×S1 + f_phase2×S2 + f_phase3×S3
```

**Approach 2: Dominant Phase**
```
Strength ≈ S_dominant_phase
```
Where dominant = highest fraction

**Approach 3: Mixture + Strengthening**
```
Strength = f_matrix×S_matrix + f_precip×Strengthening_coefficient
```

---

## Troubleshooting

### Issue: Predicted strength is negative

**Cause:** Temperature coefficient is too large or phase properties are wrong

**Solution:**
- Check that phase properties are positive
- Reduce magnitude of temperature coefficient B
- Ensure liquid phase has low but positive strength (1-10 MPa)

---

### Issue: Strength doesn't change with temperature

**Cause:** Phase fractions don't change, or coefficients are zero

**Solution:**
- Check that temperature range includes phase transitions
- Verify phase fractions are changing in the data table
- Increase temperature range or check TDB database

---

### Issue: Sharp discontinuities in curve

**Cause:** Phase transformation at specific temperature

**Solution:**
- **This is expected!** Discontinuities occur at phase boundaries
- Reduce temperature step for smoother curve
- In real alloys, transformation hysteresis smooths this out

---

### Issue: Results don't match experimental data

**Possible causes:**
1. **Wrong phase properties** → Adjust values based on literature
2. **Inaccurate TDB database** → Get better database or use ESPEI to refine
3. **Missing strengthening mechanisms** → Add grain size, precipitation effects
4. **Non-equilibrium microstructure** → CALPHAD assumes equilibrium (real alloys may not be)

**Solutions:**
- Fit empirical coefficients to your experimental data
- Use experimental comparison mode to identify discrepancies
- Consider using ESPEI to build custom database

---

## Summary

### Key Capabilities

✅ **Calculate phase fractions vs temperature** for any composition
✅ **Three prediction methods** (Rule of Mixtures, Empirical, Experimental)
✅ **Generate publication-quality plots** (strength curves + phase evolution)
✅ **Export data for ML training** (combine with Materials Project data)
✅ **Validate against experiments** (side-by-side comparison)

### Typical Workflow

1. Load TDB database
2. Define composition (e.g., Al-30%Zn)
3. Set temperature range (e.g., 300-800K)
4. Choose property model
5. Calculate (5-30 seconds)
6. View results (curves + data table)
7. Export CSV for ML or publication

### Integration Benefits

**With CALPHAD alone:**
- Understand phase stability
- No property predictions

**With CALPHAD + This Feature:**
- ✅ Phase stability + mechanical properties
- ✅ Temperature-dependent predictions
- ✅ ML-ready datasets
- ✅ Experimental validation

---

## Next Steps

**For beginners:**
1. Try Al-Zn example with Rule of Mixtures
2. Experiment with different compositions (10%, 20%, 30% Zn)
3. Compare how strength curves change

**For intermediate users:**
1. Upload your experimental data
2. Fit empirical coefficients
3. Build predictive model for your alloy system

**For advanced users:**
1. Automate calculations for many compositions
2. Export to ML training pipeline
3. Use ESPEI to refine TDB database based on experimental fits

---

**🔥 Start predicting temperature-strength curves now!**

*Part of the ML Alloy Discovery Project - CALPHAD App*
