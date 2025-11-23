# ESPEI Integration Guide

## Overview

The CALPHAD app now includes **ESPEI integration** to create TDB (Thermodynamic DataBase) files from Materials Project CSV data!

## What is ESPEI?

**ESPEI** (Extensible Self-optimizing Phase Equilibria Infrastructure) is a powerful tool for:
- Creating CALPHAD thermodynamic databases
- Generating model parameters from DFT and experimental data
- Optimizing thermodynamic parameters using Bayesian methods

## How to Use the ESPEI Integration

### Step 1: Prepare Your Data

You need a CSV file from Materials Project with:
- **Element composition columns**: `Fe`, `Cr`, `Ni`, etc. (mole fractions)
- **Formation energy column**: `formation_energy_per_atom` (in eV/atom)
- **Optional**: `material_id`, `formula` for tracking

**Example CSV:**
```csv
material_id,formula,Fe,Cr,formation_energy_per_atom
mp-123,Fe0.5Cr0.5,0.5,0.5,-0.234
mp-456,Fe0.7Cr0.3,0.7,0.3,-0.156
mp-789,Fe0.9Cr0.1,0.9,0.1,-0.045
```

### Step 2: Run the ESPEI Tool

1. **Launch the app:**
   ```bash
   streamlit run calphad_app.py --server.port 8503
   ```

2. **Navigate to:** `🔬 Create TDB from MP Data`

3. **Upload your CSV file**

4. **Configure settings:**
   - Select element columns
   - Choose formation energy column
   - Set number of materials to process (start small for testing!)
   - Choose output filename

5. **Click:** `🔥 Generate TDB File`

### Step 3: Use Your TDB File

Once generated, you can:

1. **Copy to calphad_databases folder** (use the button in the app)

2. **Load in CALPHAD app:**
   - Go to `📁 Load Database`
   - Select your new TDB file

3. **Calculate phase diagrams:**
   - `📊 Binary Phase Diagram`
   - `⚖️ Equilibrium Calculator`
   - `💪 Temperature-Property Curves`

## How It Works

### Data Conversion Process

1. **CSV Parsing:**
   - Reads Materials Project CSV data
   - Extracts composition and formation energy for each material

2. **ESPEI Dataset Creation:**
   - Converts each material into an ESPEI JSON dataset
   - Formation energy (eV/atom) → Enthalpy of formation (J/mol)
   - Creates dataset with proper ESPEI structure

3. **Phase Model Generation:**
   - Creates a simple liquid phase model
   - Can be extended to include solid phases (FCC, BCC, etc.)

4. **Parameter Generation:**
   - ESPEI fits Gibbs energy model parameters
   - Uses linear regression with AICc to prevent overfitting
   - Generates TDB file with optimized parameters

### Technical Details

**Energy Conversion:**
- Materials Project uses eV/atom
- CALPHAD uses J/mol
- Conversion: 1 eV/atom ≈ 96,485.3 J/mol

**Data Format:**
ESPEI expects JSON datasets like:
```json
{
  "components": ["CR", "FE"],
  "phases": ["LIQUID"],
  "conditions": {
    "P": 101325,
    "T": 298.15
  },
  "output": "HM_FORM",
  "values": [[[-22600.5]]],
  "reference": "mp-123"
}
```

## Limitations and Future Improvements

### Current Limitations

1. **0K DFT Data:** Materials Project data is at 0K (DFT), but we assign it to 298.15 K for CALPHAD
2. **Simple Phase Models:** Currently only creates liquid phase models
3. **Formation Energy Only:** Only uses formation energy, not other properties
4. **No Temperature Dependence:** DFT data is single-temperature

### Planned Improvements

1. **Multi-phase support:** Add FCC, BCC, HCP, etc.
2. **Temperature extrapolation:** Use empirical models to estimate T-dependence
3. **Additional properties:** Include elastic moduli, magnetic properties
4. **MCMC optimization:** Add experimental data and run Bayesian optimization
5. **Validation tools:** Compare TDB predictions with DFT data

## Advanced Usage

### Adding Experimental Data

For better databases, add experimental phase boundary data:

1. Create experimental datasets in ESPEI JSON format
2. Place in the `espei_datasets` folder
3. Run ESPEI MCMC optimization:
   ```yaml
   mcmc:
     iterations: 1000
     chains_per_parameter: 2
     chain_std_deviation: 0.1
   ```

### Refining the Database

After generating the initial TDB:

1. **Review parameters:** Check if they're physically reasonable
2. **Add more data:** Include more materials from Materials Project
3. **Validate predictions:** Compare with known phase diagrams
4. **Iterate:** Refine and regenerate as needed

## Example Workflow

### Complete Example: Fe-Cr System

1. **Download Fe-Cr alloys from Materials Project**
   - Use App 1 (Materials Project Explorer)
   - Query: Fe-Cr binary alloys
   - Download CSV

2. **Generate TDB with ESPEI**
   - Open CALPHAD app → `🔬 Create TDB from MP Data`
   - Upload Fe-Cr CSV
   - Process 20-50 materials
   - Generate `fe_cr_mp.tdb`

3. **Calculate Phase Diagram**
   - Load `fe_cr_mp.tdb`
   - Go to `📊 Binary Phase Diagram`
   - Calculate Fe-Cr phase diagram
   - Compare with experimental literature!

4. **Use for ML Training**
   - Extract CALPHAD features at various temperatures
   - Combine with MP features
   - Train better ML models!

## Troubleshooting

### Error: "No element columns detected"
- **Cause:** CSV doesn't have element composition columns
- **Fix:** Ensure columns are named `Fe`, `Cr`, `Ni`, etc. (case-insensitive)

### Error: "Formation energy not found"
- **Cause:** No formation energy column
- **Fix:** Add column named `formation_energy_per_atom` or select correct column

### Error: "ESPEI generation failed"
- **Cause:** Insufficient data or invalid values
- **Fix:**
  - Increase number of materials (need at least 3-5)
  - Check for NaN values in formation energies
  - Verify compositions sum to ≈ 1.0

### Warning: "Compositions don't sum to 1.0"
- **Cause:** Values might be in atomic % (0-100) instead of fractions (0-1)
- **Fix:** Enable "Normalize to sum = 1.0" checkbox

## Resources

- **ESPEI Documentation:** https://espei.org
- **ESPEI GitHub:** https://github.com/phasesresearchlab/espei
- **PyCalphad Docs:** https://pycalphad.org
- **CALPHAD Journal:** https://www.sciencedirect.com/journal/calphad
- **Materials Project:** https://materialsproject.org

## Citation

If you use ESPEI in your research, please cite:

```
Bocklund, B., Otis, R., Egorov, A., Obaied, A., Roslyakova, I., & Liu, Z. K. (2019).
ESPEI for efficient thermodynamic database development, modification, and uncertainty quantification: application to Cu–Mg.
MRS Communications, 9(2), 618-627.
```

## Support

For issues or questions:
- **ESPEI Issues:** https://github.com/phasesresearchlab/espei/issues
- **PyCalphad Issues:** https://github.com/pycalphad/pycalphad/issues
- **This App:** See `CALPHAD_APP_GUIDE.md`

---

**Note:** This is an experimental feature. The quality of the generated TDB files depends heavily on:
- Quality and quantity of input data
- Appropriateness of the phase models
- Validity of the 0K → 298K assumption

Always validate generated databases against known experimental data before using them for critical applications!
