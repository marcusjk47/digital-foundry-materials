# ML Alloy Development Project

Machine Learning for Novel Alloy Discovery using Materials Project Data

**Project Started:** November 6, 2025

---

## 🚀 Quick Start

### 🌐 WEB APPLICATION (RECOMMENDED!) ⭐

**Launch the interactive web dashboard - No coding required!**

1. **Double-click:** `run_webapp.bat`
2. **Your browser opens automatically** at http://localhost:8501
3. **Use the dashboard to:**
   - 📥 Download alloy data from Materials Project
   - 📊 Visualize data interactively
   - 🤖 Train ML models with point-and-click
   - 🔍 Explore composition-property relationships

**See `WEB_APP_GUIDE.md` for detailed instructions**

---

### OR: Command Line Tools

#### Activate Environment
```bash
# Windows (double-click or run from command line)
activate.bat

# Or manually
mp-alloy-env\Scripts\activate
```

#### Set API Key (if not using .env file)
```powershell
# PowerShell
$env:MP_API_KEY="b6qrDfF03oh0S6Ewo372OSqdYg3MXD6o"
```

#### Test Connection
```bash
python test_mp_connection.py
```

#### Download Data
```bash
# Quick download of Fe-Ni alloys
python download_first_dataset.py

# Full-featured downloader
python mp_data_download.py
```

---

## 📦 Environment Setup

**Python Version:** 3.13.9
**Virtual Environment:** `mp-alloy-env/`

### Installed Packages
- `mp-api` - Materials Project API client
- `pymatgen` - Materials science toolkit
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `plotly` - Interactive plots
- `scikit-learn` - Machine learning
- `tqdm` - Progress bars
- `python-dotenv` - Environment management

---

## 📁 Project Structure

```
ML-Alloy-Project/
├── mp-alloy-env/              # Virtual environment
├── .env                       # API key (DO NOT COMMIT)
├── .gitignore                 # Git ignore rules
├── activate.bat               # Easy activation script
├── test_mp_connection.py      # Test API connection
├── download_first_dataset.py  # Quick Fe-Ni download
├── mp_data_download.py        # Full downloader class
├── fe_ni_alloys.csv          # Downloaded Fe-Ni data
└── README.md                  # This file
```

---

## 📊 Downloaded Datasets

### Fe-Ni Alloys (`fe_ni_alloys.csv`)
- **Materials:** 17
- **Composition Range:** Fe₀.₂₅Ni₀.₇₅ to Fe₀.₇₅Ni₀.₂₅
- **Properties:** Formation energy, density, stability, crystal structure
- **Stable Materials:** 2 (11.8%)
- **Near-stable (E_hull < 0.1 eV):** 13 (76.5%)

---

## 🔑 API Key

Your Materials Project API key is stored in `.env` file.

**Security Notes:**
- ✅ `.env` is in `.gitignore` (will not be committed to Git)
- ⚠️ Never share your API key publicly
- ⚠️ Never commit `.env` to version control

---

## 📚 Documentation

For detailed guides, see:
- `../Project Resources/QUICK_START.md` - Getting started guide
- `../Project Resources/ML_Alloy_Development_Project_Plan.md` - Full project plan
- `../Project Resources/Materials_Project_Data_Guide.md` - API documentation

---

## 🎯 Next Steps

### Immediate
- [x] Set up Python environment
- [x] Install required packages
- [x] Get Materials Project API key
- [x] Test API connection
- [x] Download first dataset

### This Week
- [ ] Explore Fe-Ni data (Excel, Python, visualization)
- [ ] Download additional alloy systems (Ti-Al, Ni-Al, etc.)
- [ ] Start data cleaning and preparation
- [ ] Set up Jupyter notebooks for analysis

### This Month
- [ ] Complete Phase 1: Data Preparation (see project plan)
- [ ] Begin Phase 2: Model Development
- [ ] Implement baseline ML models
- [ ] Set up experiment tracking (MLflow)

---

## 💡 Usage Examples

### Load Data
```python
import pandas as pd

# Load Fe-Ni data
df = pd.read_csv('fe_ni_alloys.csv')
print(df.head())
```

### Query Materials Project
```python
from mp_api.client import MPRester
import os

api_key = os.environ.get('MP_API_KEY')

with MPRester(api_key) as mpr:
    docs = mpr.materials.summary.search(
        chemsys="Ti-Al",
        band_gap=(0, 0),  # Metallic
        fields=["material_id", "formula_pretty", "formation_energy_per_atom"]
    )
    print(f"Found {len(docs)} Ti-Al materials")
```

---

## 🔗 Resources

- **Materials Project:** https://next-gen.materialsproject.org/
- **API Docs:** https://docs.materialsproject.org/
- **pymatgen:** https://pymatgen.org/
- **Community Forum:** https://matsci.org/

---

## 📝 Notes

- Environment activated: Virtual environment isolates project dependencies
- API tested: Successfully connected to Materials Project
- First dataset: 17 Fe-Ni alloys with properties downloaded
- Ready for: Data exploration and ML model development

---

**Last Updated:** November 6, 2025
**Status:** ✅ Environment ready, API connected, data downloaded
