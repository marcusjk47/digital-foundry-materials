# ESPEI Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Launch the App
```bash
cd "C:\Users\marcu\OneDrive\Desktop\Digital Foundry\ML-Alloy-Project"
run_calphad_app.bat
```

### Step 2: Try the Example

1. **Navigate to:** `🔬 Create TDB from MP Data`

2. **Upload test file:** `test_espei_data.csv` (included in project)

3. **Configure:**
   - Elements: Fe, Cr (auto-detected ✓)
   - Formation energy: formation_energy_per_atom (auto-detected ✓)
   - Max materials: 15
   - Temperature: 298.15 K

4. **Click:** `🔥 Generate TDB File`

5. **Wait:** ESPEI will process the data (~30 seconds)

6. **Result:** You'll get `materials_project.tdb` in `espei_output/` folder

### Step 3: Use Your TDB

1. **Click:** `📋 Copy TDB to calphad_databases folder`

2. **Navigate to:** `📁 Load Database`

3. **Select:** `materials_project.tdb`

4. **Calculate:** Go to `📊 Binary Phase Diagram`
   - Element A: Fe
   - Element B: Cr
   - Calculate the Fe-Cr phase diagram!

## 📊 What Just Happened?

1. ✅ Converted 15 Fe-Cr materials from Materials Project format
2. ✅ Created ESPEI JSON datasets with formation energies
3. ✅ Generated thermodynamic model parameters
4. ✅ Produced a working TDB file for PyCalphad

## 🎯 Real-World Usage

### Get Materials Project Data

**Option 1: Use existing CSV files**
- Check your project for `Fe_Cr_alloys.csv`, `Cu_W_alloys.csv`, etc.

**Option 2: Download from Materials Project**
- Go to https://materialsproject.org
- Query for alloys (e.g., "Fe-Cr")
- Download composition and formation energy data
- Save as CSV

**Option 3: Use App 1**
- If you have the Materials Project Explorer app
- Download data for your alloy system
- Use that CSV directly

### Process Your Data

1. **Upload CSV** to the ESPEI page
2. **Verify columns:**
   - Element compositions (mole fractions)
   - Formation energies (eV/atom)
3. **Generate TDB**
4. **Use for calculations!**

## 🔧 Tips for Best Results

### Data Quality
- **More materials = Better:** Aim for 20-50+ materials
- **Coverage matters:** Try to span the full composition range
- **Check for errors:** Remove materials with obviously wrong energies

### Composition Coverage
```
Good:     Fe: 0.1 → 0.9 (many points)
Better:   Fe: 0.0 → 1.0 (complete range)
Best:     Fe: 0.0 → 1.0 (50+ evenly distributed points)
```

### Phase Models
- Current: LIQUID phase only
- Future: Add FCC, BCC, HCP, etc.
- Advanced: Edit `phase_models.json` manually

## ⚠️ Important Notes

### Limitations

1. **DFT → CALPHAD conversion** involves assumptions:
   - 0K DFT energies used at 298.15 K
   - No temperature dependence (yet)
   - Simple phase models

2. **Validation is crucial:**
   - Compare with known phase diagrams
   - Check against experimental data
   - Don't use blindly for critical applications!

3. **This is a starting point:**
   - Initial TDB for exploration
   - Refine with experimental data
   - Use ESPEI MCMC for optimization

### Best Practices

✅ **Do:**
- Start with binary systems (2 elements)
- Use test data first
- Validate predictions
- Iterate and improve

❌ **Don't:**
- Trust results blindly
- Use for critical applications without validation
- Mix incompatible data sources
- Skip the validation step!

## 🆘 Troubleshooting

### "No elements detected"
→ Check that columns are named `Fe`, `Cr`, etc. (1-2 uppercase letters)

### "Formation energy not found"
→ Ensure column is named `formation_energy_per_atom` or select manually

### "ESPEI failed"
→ Need at least 3-5 materials, check for NaN values

### "TDB doesn't work in PyCalphad"
→ Phase model might be too simple, try with more data

## 📚 Next Steps

### Learn More
- Read `ESPEI_INTEGRATION_GUIDE.md` for details
- Check ESPEI docs: https://espei.org
- PyCalphad tutorials: https://pycalphad.org

### Advanced Features
- Add experimental phase boundary data
- Run MCMC optimization
- Create multi-phase models
- Validate against literature

### Integration with ML
1. Generate TDB from DFT data
2. Calculate phase fractions at various T
3. Use as features in ML models
4. Improve property predictions!

## 🎓 Example Project

**Goal:** Predict strength of Fe-Cr alloys

1. **Get DFT data:** Download Fe-Cr from Materials Project
2. **Generate TDB:** Use ESPEI integration
3. **Calculate features:** Phase fractions vs T
4. **Train ML model:** Combine DFT + CALPHAD features
5. **Result:** Better predictions! 🎉

---

**Ready to create your first TDB file? Start with `test_espei_data.csv` and follow the steps above!**
