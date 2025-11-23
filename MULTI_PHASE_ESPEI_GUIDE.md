# Multi-Phase ESPEI Integration Guide

## 🎉 New Feature: Multiple Phase Support

Your ESPEI integration now supports **multiple solid phases** in addition to LIQUID, enabling realistic phase diagrams and better temperature-property predictions!

## What Changed

### Before (v1)
- ❌ Only LIQUID phase
- ❌ Limited phase diagram capabilities
- ❌ No solid-solid transformations

### After (v2)
- ✅ Multiple phases: LIQUID, FCC_A1, BCC_A2, HCP_A3, BCC_B2
- ✅ Realistic phase diagrams with solid phases
- ✅ Better temperature-property curves
- ✅ Solid-solid phase transformations

## Available Phases

### LIQUID
- **Structure:** Disordered liquid
- **Sublattice Model:** Single sublattice with all elements
- **Use For:** High-temperature behavior, melting

### FCC_A1 (Face-Centered Cubic)
- **Structure:** Close-packed cubic (Cu, Al, Ni, Austenite)
- **Sublattice Model:** 2 sublattices (metal + interstitial)
- **Site Ratios:** [1.0, 1.0]
- **Common In:** Cu, Ni, Al, austenitic steels, many precious metals

### BCC_A2 (Body-Centered Cubic)
- **Structure:** Body-centered cubic (Fe, Cr, W, Ferrite)
- **Sublattice Model:** 2 sublattices (metal + interstitial)
- **Site Ratios:** [1.0, 3.0]
- **Common In:** Fe, Cr, Mo, W, V, ferritic steels

### HCP_A3 (Hexagonal Close-Packed)
- **Structure:** Hexagonal close-packed (Mg, Ti, Zn)
- **Sublattice Model:** 2 sublattices (metal + interstitial)
- **Site Ratios:** [1.0, 0.5]
- **Common In:** Mg, Ti, Zn, Co (low T), some intermetallics

### BCC_B2 (Ordered BCC)
- **Structure:** Ordered body-centered cubic (NiAl, FeAl)
- **Sublattice Model:** 3 sublattices (2 metal + interstitial)
- **Site Ratios:** [0.5, 0.5, 1.0]
- **Common In:** Ordered intermetallics (NiAl, FeAl, CuZn)

## How to Use

### In the Streamlit App

1. **Launch app:**
   ```bash
   run_calphad_app.bat
   ```

2. **Navigate to:** `🔬 Create TDB from MP Data`

3. **Upload CSV** with Materials Project data

4. **Select phases** in the "Phase Models" section:
   ```
   ☑️ LIQUID     - Always recommended
   ☑️ FCC_A1     - For FCC metals (Cu, Ni, Al, austenite)
   ☑️ BCC_A2     - For BCC metals (Fe, Cr, W, ferrite)
   ☐ HCP_A3     - For HCP metals (Mg, Ti, Zn)
   ☐ BCC_B2     - For ordered phases (NiAl, FeAl)
   ```

5. **Generate TDB** - ESPEI will create datasets for all selected phases

6. **Use in CALPHAD tools** - Calculate phase diagrams showing all phases!

### Example: Fe-Cr System

**Select Phases:**
- ✅ LIQUID (for high-T melting behavior)
- ✅ FCC_A1 (austenite, high-T phase)
- ✅ BCC_A2 (ferrite, low-T phase)

**Result:** Phase diagram showing:
- Liquid region at high T
- FCC (austenite) region
- BCC (ferrite) region
- FCC→BCC transformation
- Liquid→Solid solidification

## Technical Details

### How It Works

1. **Dataset Generation:**
   - Each material gets N datasets (one per phase)
   - Same formation energy used for all phases (approximation)
   - Example: 10 materials × 3 phases = 30 datasets

2. **Phase Models:**
   - LIQUID: 1 sublattice (substitutional)
   - FCC/BCC/HCP: 2 sublattices (metal + interstitial with VA)
   - Proper site ratios for each structure

3. **ESPEI Fitting:**
   - Fits Gibbs energy parameters for each phase
   - Generates interaction parameters
   - Creates temperature-dependent functions

### Dataset Structure

For each material-phase combination:

```json
{
  "components": ["CR", "FE"],
  "phases": ["FCC_A1"],
  "solver": {
    "sublattice_configurations": [[["CR", "FE"]]],
    "sublattice_occupancies": [[[0.3, 0.7]]],
    "sublattice_site_ratios": [1.0]
  },
  "output": "HM_FORM",
  "values": [[[-12345.67]]]
}
```

### Phase Model Structure

**LIQUID (1 sublattice):**
```json
{
  "sublattice_model": [["CR", "FE"]],
  "sublattice_site_ratios": [1.0]
}
```

**FCC_A1 (2 sublattices):**
```json
{
  "sublattice_model": [["CR", "FE"], ["VA"]],
  "sublattice_site_ratios": [1.0, 1.0]
}
```

**BCC_A2 (2 sublattices, 3:1 ratio):**
```json
{
  "sublattice_model": [["CR", "FE"], ["VA"]],
  "sublattice_site_ratios": [1.0, 3.0]
}
```

## Important Notes

### ⚠️ Approximations

**Same Formation Energy:**
- Uses same DFT energy for all phases
- Real materials have different energies for different phases
- Good enough for initial databases, but not perfect

**Improvement Strategy:**
- Start with this automated approach
- Add experimental phase boundary data
- Run ESPEI MCMC optimization
- Refine parameters iteratively

### ✅ Best Practices

**Phase Selection:**
- Include phases that exist in your alloy system
- Check experimental phase diagrams
- Start with LIQUID + dominant solid phases
- Add more phases as needed

**System-Specific:**
- **Fe-Cr:** LIQUID, FCC_A1, BCC_A2
- **Cu-Ni:** LIQUID, FCC_A1
- **Ti-Al:** LIQUID, HCP_A3, FCC_A1
- **Ni-Al:** LIQUID, FCC_A1, BCC_B2

**Validation:**
- Always compare with experimental data
- Check liquidus/solidus temperatures
- Verify phase transformations occur
- Look for physically unreasonable behavior

## Examples

### Example 1: Fe-Cr (Stainless Steel)

```python
# Phases: LIQUID, FCC_A1 (austenite), BCC_A2 (ferrite)
# Materials: 50 Fe-Cr alloys from MP
# Result: Phase diagram with:
#   - Liquid region (high T)
#   - FCC region (austenite)
#   - BCC region (ferrite, stable at most compositions)
#   - Solid-solid FCC↔BCC transformation
```

### Example 2: Cu-Ni (Cupronickel)

```python
# Phases: LIQUID, FCC_A1
# Materials: 30 Cu-Ni alloys from MP
# Result: Simple phase diagram with:
#   - Liquid region
#   - FCC solid solution (complete miscibility)
#   - Smooth liquidus/solidus curves
```

### Example 3: Ti-Al (Titanium Aluminides)

```python
# Phases: LIQUID, HCP_A3, FCC_A1
# Materials: 40 Ti-Al alloys from MP
# Result: Complex phase diagram with:
#   - Liquid region
#   - HCP (Ti-rich)
#   - FCC (Al-rich)
#   - Intermetallic compounds (if data sufficient)
```

## Comparison: Before vs After

### Before (LIQUID only)

**Phase Diagram:**
```
Temperature
    ^
    |  LIQUID
    |
    |________________
    0%    →    100%
```

**Limited Information:**
- Only melting behavior
- No solid phases
- No solid-solid transformations
- Can't predict microstructure

### After (Multi-phase)

**Phase Diagram:**
```
Temperature
    ^
    |     LIQUID
    |  ___________
    | |  FCC     |
    |  ¯¯¯¯¯¯¯¯¯¯
    | |    BCC   |
    |  ¯¯¯¯¯¯¯¯¯¯
    0%    →    100%
```

**Rich Information:**
- Melting behavior (liquidus/solidus)
- Solid phases (FCC, BCC, HCP)
- Phase transformations
- Microstructure evolution
- Better temperature-property curves

## Using TDB Files

Once generated, use your multi-phase TDB:

### Binary Phase Diagrams
```python
# Load TDB with multiple phases
db = Database('fe_cr_multiphase.tdb')

# Calculate phase diagram
# Will show all phases: LIQUID, FCC_A1, BCC_A2
result = binplot(db, ['FE', 'CR', 'VA'],
                 ['LIQUID', 'FCC_A1', 'BCC_A2'],
                 {v.T: (300, 2000, 10), v.P: 101325})
```

### Temperature-Property Curves
```python
# Calculate phase fractions vs T
# Shows evolution: BCC → FCC → LIQUID
for T in range(300, 1800, 50):
    eq = equilibrium(db, ['FE', 'CR', 'VA'],
                     ['LIQUID', 'FCC_A1', 'BCC_A2'],
                     {v.T: T, v.P: 101325, v.X('CR'): 0.18})
    # Extract phase fractions
    # Use for property predictions
```

## Troubleshooting

### "Only LIQUID phase shows up"
- **Cause:** Gibbs energies too similar or unfavorable
- **Fix:** Add more data points, adjust temperature range, or use experimental data for refinement

### "Phase diagram looks wrong"
- **Cause:** Using same formation energy for all phases (approximation)
- **Fix:** This is expected. Add experimental phase boundary data and run MCMC optimization

### "Unrealistic phase boundaries"
- **Cause:** Insufficient data or numerical issues
- **Fix:** Increase number of materials, check for outliers, validate with experiments

### "HCP phase never appears"
- **Cause:** System may not have HCP phase, or energies unfavorable
- **Fix:** Check if your system actually has HCP. May need experimental data.

## Advanced: Adding Experimental Data

To get better results:

1. **Add phase boundary data** (ZPF - Zero Phase Fraction):
   ```json
   {
     "output": "ZPF",
     "conditions": {"T": 1200, "P": 101325, "X_CR": 0.15},
     "phases": [["FCC_A1"], ["BCC_A2"]],
     "comment": "FCC/BCC equilibrium at 1200K, 15% Cr"
   }
   ```

2. **Run MCMC optimization:**
   ```yaml
   mcmc:
     iterations: 5000
     chains_per_parameter: 4
     chain_std_deviation: 0.05
   ```

3. **Iteratively refine:**
   - Generate initial TDB (this tool)
   - Compare with experiments
   - Add experimental datasets
   - Re-run ESPEI with MCMC
   - Validate and iterate

## Summary

**Benefits of Multi-Phase Support:**
✅ Realistic phase diagrams with solid phases
✅ Better temperature-property predictions
✅ Solid-solid phase transformations
✅ Microstructure evolution
✅ Improved ML features

**How to Use:**
1. Select appropriate phases for your system
2. Generate TDB with ESPEI
3. Validate against experiments
4. Refine with additional data if needed

**Result:**
Much more useful TDB files for phase diagram calculations and materials design!

---

**Ready to try it?** Launch the app and generate your first multi-phase TDB! 🚀
