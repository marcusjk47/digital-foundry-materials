# 🔧 Materials Project Data - Troubleshooting Guide

**Quick fixes for common MP data import issues**

---

## Error: "Could not extract composition from selected material"

### What This Means

The CALPHAD app couldn't find composition columns in your MP CSV that match the database elements.

**Example:**
- Database has elements: `AL`, `ZN`
- CSV has columns: `aluminum_fraction`, `zinc_fraction`
- ❌ **No match** → Error!

---

## Quick Fix Steps

### Step 1: Check the Debug Information

When you see the error, **click the "🔍 Debug Information" expander** that appears.

You'll see:
```
Database elements (looking for):
AL, ZN

Available columns in MP data:
material_id, formula, Al, Zn, formation_energy, density

Non-zero numeric columns in this row:
Al=0.7, Zn=0.3, formation_energy=-0.05, density=4.2
```

**Analysis:**
- Database wants: `AL`, `ZN` (uppercase)
- CSV has: `Al`, `Zn` (title case)
- ✅ **This should work!** (case-insensitive matching)

---

### Step 2: Identify the Problem

**Common scenarios:**

#### Scenario A: Different Column Names
```
Database wants: FE, CR
CSV has: iron, chromium
```
**Solution:** Rename CSV columns to `FE`, `CR` (or `Fe`, `Cr`)

#### Scenario B: Missing Composition Columns
```
Database wants: FE, CR
CSV has: material_id, formula, formation_energy
```
**Solution:** CSV needs `FE` and `CR` columns with mole fractions

#### Scenario C: Wrong Database Loaded
```
Database wants: AL, ZN
CSV has: Fe, Cr
```
**Solution:** Load correct TDB database (Fe-Cr instead of Al-Zn)

---

## Solutions

### Solution 1: Rename Columns in CSV

**If your CSV has composition data but wrong names:**

**Example CSV (before):**
```csv
material_id,aluminum,zinc,energy
mp-123,0.7,0.3,-0.05
```

**Fix in Excel/Python:**
```python
import pandas as pd

df = pd.read_csv('materials.csv')
df = df.rename(columns={
    'aluminum': 'Al',
    'zinc': 'Zn'
})
df.to_csv('materials_fixed.csv', index=False)
```

**Example CSV (after):**
```csv
material_id,Al,Zn,energy
mp-123,0.7,0.3,-0.05
```

✅ Now it works!

---

### Solution 2: Download with Composition Columns

**When downloading from Materials Project (App 1):**

Make sure your download includes element columns!

**App 1 → Download Data:**
1. Select elements (e.g., Fe, Cr)
2. Click Download
3. **Check CSV** - should have `Fe` and `Cr` columns

**What the CSV should look like:**
```csv
material_id,formula,Fe,Cr,Ni,formation_energy
mp-123,FeCr,0.5,0.5,0.0,-0.045
mp-456,Fe2Cr,0.67,0.33,0.0,-0.032
```

**Key columns needed:**
- `material_id` (optional, for display)
- `formula` (optional, for display)
- **Element columns** (required): `Fe`, `Cr`, etc. with mole fractions

---

### Solution 3: Load Matching Database

**Make sure your TDB database matches your data!**

**Example:**
- CSV has Fe-Cr alloys (Fe, Cr columns)
- Load Fe-Cr TDB database (not Al-Zn!)

**Steps:**
1. Check what elements are in your CSV
2. Go to "Load Database" page
3. Load TDB with those same elements
4. Return to Equilibrium Calculator/Temperature-Property page

---

### Solution 4: Use Formula Parser (Future Feature)

Currently, the app requires explicit composition columns. In a future update, we may add formula parsing (e.g., "Fe0.5Cr0.5" → Fe=0.5, Cr=0.5).

**Workaround for now:**
Parse formulas in Python before uploading:

```python
import pandas as pd
from pymatgen.core import Composition

df = pd.read_csv('materials.csv')

# Extract composition from formula
for idx, row in df.iterrows():
    comp = Composition(row['formula'])
    comp_dict = comp.get_el_amt_dict()

    # Normalize to fractions
    total = sum(comp_dict.values())
    for elem, amt in comp_dict.items():
        df.at[idx, elem] = amt / total

df.to_csv('materials_with_composition.csv', index=False)
```

---

## Verification Checklist

Before uploading CSV, verify:

- [ ] CSV has columns matching database elements (e.g., `Fe`, `Cr`)
- [ ] Column names are element symbols (not full names like "iron")
- [ ] Values are mole fractions between 0 and 1
- [ ] Values sum to approximately 1.0 per row
- [ ] TDB database loaded has same elements as CSV

**Example good CSV:**
```csv
material_id,formula,Fe,Cr,formation_energy
mp-123,Fe0.5Cr0.5,0.5,0.5,-0.045
mp-456,Fe0.7Cr0.3,0.7,0.3,-0.032
```

**Verification:**
- ✅ Has `Fe` and `Cr` columns
- ✅ Names are element symbols
- ✅ Values between 0-1
- ✅ 0.5 + 0.5 = 1.0 ✓
- ✅ Loaded Fe-Cr TDB database

---

## Common Questions

### Q: Do column names need to match case exactly?

**A:** No! The app now uses **case-insensitive** matching.

These all work:
- `AL`, `ZN` (uppercase)
- `Al`, `Zn` (title case)
- `al`, `zn` (lowercase)

### Q: Can I use weight fractions instead of mole fractions?

**A:** No, CALPHAD requires **mole fractions**.

Convert weight % to mole fractions:
```python
from pymatgen.core import Composition

# Example: 50 wt% Fe, 50 wt% Cr
comp = Composition({'Fe': 50, 'Cr': 50})  # weight percent
comp_dict = comp.get_el_amt_dict()  # converts to mole fractions

print(comp_dict)  # {'Fe': 0.52, 'Cr': 0.48}
```

### Q: What if my composition doesn't sum to 1.0?

**A:** The app **automatically normalizes** compositions.

Example:
- CSV has: Fe=50, Cr=30 (sums to 80)
- App converts to: Fe=0.625, Cr=0.375 (sums to 1.0)

### Q: Can I use ternary or higher systems?

**A:** Yes! Just make sure:
- CSV has all element columns
- Database includes all elements
- System type matches (select "Ternary" for 3 elements)

---

## Debug Examples

### Example 1: Working Case

**Debug info shows:**
```
Database elements: AL, ZN
Available columns: material_id, formula, Al, Zn, formation_energy
Non-zero numeric columns: Al=0.7, Zn=0.3
```

**Analysis:** ✅ Perfect match!
- Database wants `AL`, `ZN`
- CSV has `Al`, `Zn` (case-insensitive match works)
- Both have values

---

### Example 2: Missing Columns

**Debug info shows:**
```
Database elements: FE, CR
Available columns: material_id, formula, formation_energy, band_gap
Non-zero numeric columns: formation_energy=-0.05, band_gap=0.2
```

**Analysis:** ❌ No composition columns!
- Database wants `FE`, `CR`
- CSV doesn't have `Fe` or `Cr` columns
- Only has MP properties

**Solution:** Re-download CSV from App 1 with element columns included

---

### Example 3: Wrong Database

**Debug info shows:**
```
Database elements: AL, ZN
Available columns: material_id, formula, Fe, Cr, formation_energy
Non-zero numeric columns: Fe=0.5, Cr=0.5
```

**Analysis:** ❌ Database mismatch!
- Database is for Al-Zn
- CSV has Fe-Cr data

**Solution:** Load Fe-Cr TDB database instead

---

## Best Practices

### When Downloading from Materials Project

**In App 1, make sure to:**
1. Select specific elements (not just "all materials")
2. Verify downloaded CSV has element columns
3. Check first few rows have composition values

**Good download:**
```
Elements selected: Fe, Cr
CSV columns: material_id, formula, Fe, Cr, formation_energy, ...
First row: mp-123, FeCr, 0.5, 0.5, -0.045, ...
```

### When Uploading to CALPHAD App

**Check before uploading:**
1. Open CSV in Excel/text editor
2. Verify element columns exist
3. Verify values are between 0-1
4. Note which elements are present

**Then in CALPHAD app:**
1. Load TDB database with same elements
2. Upload CSV to sidebar
3. Check "Columns in CSV" expander
4. Verify elements match database

---

## Still Having Issues?

### Diagnostic Steps

1. **Check sidebar:**
   - "✅ X materials loaded"
   - Click "📋 Columns in CSV" expander
   - Note column names

2. **Check database:**
   - Go to "Load Database" page
   - Look at "Elements" list
   - Note element names

3. **Compare:**
   - Database elements: `AL`, `ZN`
   - CSV columns: `Al`, `Zn`
   - Should match (case-insensitive)

4. **Try simple test:**
   - Create minimal CSV:
     ```csv
     Al,Zn
     0.7,0.3
     0.5,0.5
     ```
   - Load alzn_sample.tdb database
   - Upload CSV
   - Try selecting a material

---

## Quick Reference

| Problem | Solution |
|---------|----------|
| Column names don't match | Rename CSV columns to element symbols |
| Missing composition columns | Re-download from App 1 with elements selected |
| Wrong database loaded | Load TDB matching CSV elements |
| Values not 0-1 | Convert to mole fractions |
| Composition doesn't sum to 1 | App auto-normalizes (no action needed) |

---

## Example: Complete Fix

**Problem:**
```
Error: Could not extract composition from selected material
```

**Debug info:**
```
Database elements: FE, CR
Available columns: iron_frac, chromium_frac, energy
Non-zero: iron_frac=0.7, chromium_frac=0.3
```

**Solution:**
```python
# Fix CSV column names
import pandas as pd
df = pd.read_csv('materials.csv')
df = df.rename(columns={'iron_frac': 'Fe', 'chromium_frac': 'Cr'})
df.to_csv('materials_fixed.csv', index=False)
```

**Result:**
```
✅ Composition from MP: Fe=0.700, Cr=0.300
```

Success! 🎉

---

**Still stuck? Check that:**
1. ✅ CSV has element symbol columns (`Fe`, `Cr`, etc.)
2. ✅ TDB database has same elements
3. ✅ Values are numbers between 0 and 1
4. ✅ At least one composition per row is non-zero

*ML Alloy Discovery Project - Troubleshooting Guide*
