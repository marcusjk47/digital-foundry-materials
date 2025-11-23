# 📊 Using Materials Project Data in CALPHAD Features

**Quick guide for seamlessly using MP compositions across all CALPHAD tools**

---

## New Feature: Global MP Data Loader

Materials Project data can now be loaded **once** in the sidebar and used across **all** CALPHAD features!

**No more manual composition entry - just select from your MP data!**

---

## How It Works

### Step 1: Load MP Data (Sidebar)

**In the CALPHAD app sidebar:**
1. Look for the "📊 Materials Project Data" section (below navigation menu)
2. Click "Browse files" and upload your MP CSV
3. See success message: "✅ X materials loaded"

**Data stays loaded as you navigate between pages!**

---

### Step 2: Use MP Compositions

**Available on these pages:**
- ⚖️ **Equilibrium Calculator**
- 💪 **Temperature-Property Curves**

**How to use:**
1. Check the box: "📊 Use composition from Materials Project data"
2. Select a material from the dropdown (shows material_id and formula)
3. Composition automatically filled in!
4. Calculate normally

---

## Complete Example Workflow

### Goal: Calculate Phase Diagram & Properties for Fe-Cr Alloys

**Step 1: Download from Materials Project (App 1)**
```
App 1 → Download Data → Fe-Cr system → fe_cr_alloys.csv (50 materials)
```

**Step 2: Load in CALPHAD App (App 3 Sidebar)**
```
App 3 → Sidebar → Upload fe_cr_alloys.csv
✅ 50 materials loaded
```

**Step 3: View Full Phase Diagram**
```
App 3 → 📊 Binary Phase Diagram → Select Fe, Cr → Calculate
Result: Full Fe-Cr phase diagram (0-100% composition)
```

**Step 4: Analyze Specific Composition from MP**
```
App 3 → ⚖️ Equilibrium Calculator
☑ Use composition from Materials Project data
Select: "mp-123 - Fe0.7Cr0.3"
✅ Composition from MP: Fe=0.700, Cr=0.300
Temperature: 1000K
Calculate → See phases at this exact MP composition!
```

**Step 5: Temperature-Strength Curve for Same Composition**
```
App 3 → 💪 Temperature-Property Curves
☑ Use composition from Materials Project data
Select: "mp-123 - Fe0.7Cr0.3"
✅ Composition from MP: Fe=0.700, Cr=0.300
Temperature range: 300-1500K
Property Model: Rule of Mixtures
Calculate → Strength vs temperature for this MP material!
```

**Step 6: Batch Process All Materials**
```
App 3 → 🔗 Batch MP Analysis
Upload: fe_cr_alloys.csv (automatically processed)
Calculate at: 1000K
Export: fe_cr_with_calphad.csv
→ All 50 materials get CALPHAD features!
```

---

## Use Cases

### Use Case 1: Deep Dive on One Material

**Scenario:** MP says Fe-50%Cr (mp-456) has low formation energy. What's the full story?

**Workflow:**
1. Upload MP data (sidebar)
2. **Binary Phase Diagram:** See where Fe-50%Cr sits on full diagram
3. **Equilibrium Calculator:** What phases at 800K? → 45% FCC, 55% BCC
4. **Temperature-Property:** How does strength vary? → Strong up to 900K
5. **Result:** Complete thermodynamic + property profile!

---

### Use Case 2: Compare Multiple Compositions

**Scenario:** Compare 3 Fe-Cr compositions from MP

**Workflow:**
1. Upload MP data with Fe-30%Cr, Fe-50%Cr, Fe-70%Cr
2. **Equilibrium Calculator:**
   - Select Fe-30%Cr → Calculate phases at 1000K
   - Select Fe-50%Cr → Calculate phases at 1000K
   - Select Fe-70%Cr → Calculate phases at 1000K
3. **Compare results:**
   - Fe-30%Cr: Mostly FCC
   - Fe-50%Cr: Mixed FCC+BCC
   - Fe-70%Cr: Mostly BCC
4. **Conclusion:** Composition controls microstructure!

---

### Use Case 3: Validate MP Predictions

**Scenario:** Does CALPHAD agree with MP about stability?

**For each MP material:**
1. Check MP formation_energy (0K stability)
2. Calculate CALPHAD Gibbs energy at 0K (extrapolate)
3. Compare predicted phases
4. Identify discrepancies
5. → Database validation!

---

## Features & Benefits

### Auto-Detection
- Automatically finds composition columns (Fe, Cr, Ni, etc.)
- Normalizes to sum = 1.0
- Handles different data formats

### Smart Display
- Shows material_id + formula: "mp-123 - Fe0.5Cr0.5"
- Or formula only if no material_id
- Clear confirmation of selected composition

### Persistent Loading
- Load data once, use everywhere
- Stays in memory as you navigate pages
- "Using previously loaded data" message

### Error Handling
- Checks for valid composition
- Warns if missing elements
- Graceful fallback to manual entry

---

## Tips & Tricks

### Tip 1: Prepare MP Data

**Best CSV format:**
```csv
material_id,formula,Fe,Cr,Ni,formation_energy
mp-123,Fe0.5Cr0.5,0.5,0.5,0.0,-0.045
mp-456,Fe0.7Cr0.3,0.7,0.3,0.0,-0.032
```

**Make sure:**
- Composition columns match database elements
- Values are mole fractions (0.0-1.0)
- Include material_id and formula for easy identification

---

### Tip 2: Batch + Individual Analysis

**Strategy:**
1. **Batch MP Analysis:** Process all 50 materials (get overview)
2. **Temperature-Property:** Select interesting ones for detailed curves
3. **Equilibrium Calculator:** Deep dive on outliers

**Example:**
- Batch shows mp-456 has unusual phase fractions
- Temperature curve shows strength drops suddenly at 800K
- Equilibrium at 800K reveals phase transformation
- → Understanding gained!

---

### Tip 3: Cross-Reference with Phase Diagram

**Workflow:**
1. Calculate full binary phase diagram (Fe-Cr)
2. Note where MP compositions sit on diagram
3. Use composition selector to verify predictions
4. → Visual + quantitative understanding

---

### Tip 4: Use for ML Feature Generation

**Combine with Batch Analysis:**
1. Upload MP CSV to sidebar
2. Use Equilibrium Calculator to spot-check a few
3. Once confident, go to Batch MP Analysis
4. Process all → Export ML-ready dataset
5. Train in App 1!

---

## Troubleshooting

### Issue: "Could not extract composition"

**Cause:** Column names don't match database elements

**Solution:**
- Check sidebar: "Elements in database: AL, ZN"
- Check CSV columns: Should have "AL" and "ZN" columns
- Rename columns if needed (e.g., "Al" → "AL")

---

### Issue: Composition shows NaN or weird values

**Cause:** Missing data or wrong column format

**Solution:**
- Check CSV has composition columns with numeric values
- Verify values are between 0 and 1
- Check for NaN or empty cells

---

### Issue: Can't see material_id in dropdown

**Cause:** CSV doesn't have material_id column

**Solution:**
- App falls back to showing "Material 0", "Material 1", etc.
- Add material_id column to CSV for better identification
- Or use formula column

---

## Comparison: Before vs After

### Before (Manual Entry)
```
1. Go to Equilibrium Calculator
2. Manually select: Component A = Fe, Component B = Cr
3. Manually set slider: X(Cr) = 0.3
4. Calculate
5. Write down results
6. Repeat for next composition (go back to step 2)
```
**Time:** ~2 minutes per composition
**50 compositions:** 100 minutes (1.7 hours)

### After (MP Data Selector)
```
1. Upload MP CSV to sidebar (one time)
2. Go to Equilibrium Calculator
3. Check "Use composition from Materials Project"
4. Select "mp-123 - Fe0.7Cr0.3" from dropdown
5. Calculate
6. Select next from dropdown
```
**Time:** ~30 seconds per composition
**50 compositions:** 25 minutes
**Savings:** 75 minutes! (75% faster)

---

## Summary

**New capabilities:**
✅ Upload MP CSV **once** in sidebar
✅ Select compositions from dropdown in Equilibrium Calculator
✅ Select compositions from dropdown in Temperature-Property Curves
✅ Auto-filled composition values
✅ Persistent data across pages
✅ Much faster workflow

**Workflow improvements:**
- **No manual composition entry**
- **No calculation needed** to convert formulas
- **No risk of typos** in composition values
- **Easy comparison** of multiple materials
- **Seamless integration** with Batch MP Analysis

**Time savings:**
- Manual entry: ~2 min per composition
- MP selector: ~30 sec per composition
- **75% time savings!**

---

## Quick Reference

### Load MP Data
**Location:** Sidebar → "📊 Materials Project Data" → Upload CSV

### Use in Equilibrium Calculator
**Location:** ⚖️ Equilibrium Calculator → Step 1 → ☑ Use composition from MP → Select material

### Use in Temperature-Property Curves
**Location:** 💪 Temperature-Property Curves → Step 1 → ☑ Use composition from MP → Select material

### Batch Process
**Location:** 🔗 Batch MP Analysis → Upload CSV → Calculate

---

**🎉 Enjoy seamless MP-CALPHAD integration!**

*ML Alloy Discovery Project - CALPHAD App*
