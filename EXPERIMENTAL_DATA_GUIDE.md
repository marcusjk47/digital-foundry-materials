# 🌡️ Experimental Data Manager Guide

**Track temperature-dependent properties like Strength vs Temperature curves!**

---

## 🎯 What Is This?

A dedicated web application for managing **experimental** temperature-dependent alloy properties that aren't available in Materials Project:

- **Strength vs Temperature** curves
- **Hardness vs Temperature** curves
- **Ductility, Creep, Fatigue** data
- **Thermal properties** (conductivity, expansion, etc.)
- Any other **property that changes with temperature**

---

## 🚀 Quick Start

### Launch the App

**Double-click:** `run_experimental_app.bat`

The app opens at: **http://localhost:8502** (different port from main app!)

**You can run BOTH apps simultaneously!**
- Main app: http://localhost:8501
- Experimental app: http://localhost:8502

---

## 📊 Four Main Features

### 1️⃣ **Upload Data** 📤

Upload your experimental temperature-dependent data:

**What you need:**
- CSV file with Temperature and Property Value columns
- Material name (e.g., "Ti-6Al-4V")
- Property type (e.g., "Yield Strength")
- Units (e.g., "MPa", "HV", "GPa")

**How to do it:**
1. Click "Upload Data" in sidebar
2. Fill in material information
3. Upload your CSV file
4. Map columns (Temperature, Value)
5. Preview the curve
6. Click "Save Data"

**CSV Format Example:**
```csv
Temperature,Yield_Strength
20,950
100,920
200,880
300,830
400,750
```

---

### 2️⃣ **Visualize Curves** 📊

View individual property vs temperature curves:

**Features:**
- Interactive plots (hover, zoom, pan)
- Adjustable plot styles
- Temperature range filtering
- Statistics (min, max, average)
- Data table view
- Download filtered data

**What you can do:**
- Zoom into specific temperature ranges
- Find temperature at maximum property
- Export high-quality plots
- View full data table

---

### 3️⃣ **Compare Materials** 📈

Compare multiple materials side-by-side:

**Features:**
- Overlay up to 5 materials
- Interactive comparison plots
- Property comparison at specific temperature
- Automatic best performer identification

**Perfect for:**
- Material selection
- Trade-off analysis
- Performance benchmarking
- Presenting to stakeholders

**Example:**
Compare Ti-6Al-4V, Inconel 718, and 316 Stainless Steel strength curves!

---

### 4️⃣ **Manage Data** 💾

View, organize, and delete your datasets:

**What you can do:**
- See all uploaded datasets
- View metadata (material, property, source)
- Preview data
- Delete unwanted datasets
- Track data sources and notes

---

## 💡 Use Cases

### 1. **Aerospace Alloy Selection**
- Upload: Ti-6Al-4V strength from 20°C to 600°C
- Compare: Against Inconel 718 and aluminum alloys
- Find: Best material for 400°C operating temperature

### 2. **High-Temperature Applications**
- Upload: Creep rate vs temperature
- Visualize: Safe operating range
- Compare: Multiple superalloys

### 3. **Research Documentation**
- Upload: Experimental test results
- Visualize: Professional publication-quality plots
- Manage: All test data in one place

### 4. **Material Database**
- Upload: Properties from multiple sources
- Compare: Literature vs your lab results
- Track: Data provenance and notes

---

## 📁 Supported Properties

### **Mechanical Properties:**
- Yield Strength
- Tensile Strength
- Hardness (Vickers, Rockwell, Brinell)
- Ductility (Elongation %)
- Elastic Modulus
- Creep Rate
- Fatigue Strength

### **Thermal Properties:**
- Thermal Conductivity
- Thermal Expansion Coefficient
- Specific Heat Capacity

### **Custom Properties:**
- Define your own!

---

## 📋 CSV File Requirements

### **Minimum Requirements:**
1. Two columns: Temperature, Property Value
2. Numeric data only
3. CSV format (.csv file)

### **Column Names:**
- Can be anything (you'll map them in the app)
- Examples: `Temp_C`, `Temperature`, `T_degC`
- Examples: `Strength_MPa`, `Yield`, `Property`

### **Temperature Units:**
- °C (Celsius)
- K (Kelvin)
- °F (Fahrenheit)

### **Data Points:**
- Minimum: 3 points
- Recommended: 5-10 points
- More points = smoother curves!

---

## 🎓 Tutorial: Complete Workflow

### **Step 1: Prepare Your Data**

Create a CSV file (in Excel, Google Sheets, or text editor):

```csv
Temperature,Yield_Strength
20,950
100,920
200,880
300,830
400,750
500,650
600,500
```

Save as: `ti64_strength.csv`

---

### **Step 2: Launch the App**

Double-click: `run_experimental_app.bat`

---

### **Step 3: Upload Data**

1. Click **"Upload Data"** in sidebar
2. Fill in:
   - **Material Name:** Ti-6Al-4V
   - **Composition:** Ti-6Al-4V
   - **Property:** Yield Strength
   - **Property Unit:** MPa
   - **Temperature Unit:** °C
   - **Data Source:** ASTM Handbook (optional)
3. Click **"Choose CSV file"** → Select `ti64_strength.csv`
4. Map columns:
   - **Temperature Column:** Temperature
   - **Property Value Column:** Yield_Strength
5. Preview the plot
6. Click **"Save Data"** 💾

---

### **Step 4: Visualize**

1. Click **"Visualize Curves"** in sidebar
2. Select your dataset from dropdown
3. Adjust plot options:
   - Plot style: Lines + Markers
   - Show Grid: ✅
   - Temperature Range: adjust slider
4. View statistics
5. Download plot or data

---

### **Step 5: Compare (if you have multiple materials)**

1. Upload more materials (repeat Step 3)
2. Click **"Compare Materials"** in sidebar
3. Select materials to compare (up to 5)
4. View overlay plot
5. Select temperature for comparison table
6. Find best performer! 🏆

---

## 📊 Example Data Included

I've created 3 example datasets for you in the `example_data/` folder:

1. **Ti-6Al-4V** - Yield Strength vs Temperature
2. **Inconel 718** - Tensile Strength vs Temperature
3. **316 Stainless Steel** - Hardness vs Temperature

**Try uploading these to get started!**

---

## 🎨 Pro Tips

### **Better Visualizations:**
- Use consistent temperature ranges for comparison
- Include data points at critical temperatures
- Add notes about testing conditions

### **Data Organization:**
- Use clear, descriptive material names
- Include composition details
- Track data sources for traceability
- Add testing conditions in notes

### **Comparison:**
- Compare same property across materials
- Use same temperature units
- Normalize to room temperature for better comparison

### **Export:**
- Download plots as PNG for presentations
- Export filtered data for further analysis
- Save metadata for documentation

---

## 🔄 Integration with Main App

### **Workflow:**

1. **Materials Project (Main App):**
   - Download computational data (DFT, 0K properties)
   - Formation energy, stability, density
   - Structure, composition

2. **Experimental Data Manager:**
   - Upload real experimental data
   - Temperature-dependent properties
   - Measured strength, hardness, etc.

3. **Combined Analysis:**
   - Use MP data to screen candidates
   - Use experimental data to validate
   - Make informed decisions!

---

## ❓ FAQ

### **Q: Can I use data from literature?**
**A:** Yes! Just cite the source in the "Data Source" field.

### **Q: What if my CSV has more columns?**
**A:** No problem! You select which columns to use during upload.

### **Q: Can I update existing data?**
**A:** Delete the old dataset and upload a new one with updated values.

### **Q: How do I share my data with colleagues?**
**A:** Export the CSV files from the `experimental_data/` folder.

### **Q: Can I plot multiple properties for one material?**
**A:** Yes! Upload separate files for each property (e.g., strength, hardness).

### **Q: What's the maximum number of data points?**
**A:** No limit! The app handles large datasets efficiently.

---

## 🗂️ File Structure

```
ML-Alloy-Project/
├── experimental_data/          # Your uploaded data goes here
│   ├── Ti-6Al-4V_Yield_Strength.csv
│   ├── Ti-6Al-4V_Yield_Strength_meta.json
│   ├── Inconel718_Tensile_Strength.csv
│   └── ...
├── example_data/               # Example datasets provided
│   ├── ti64_yield_strength.csv
│   ├── inconel718_tensile_strength.csv
│   └── 316_stainless_hardness.csv
├── experimental_data_manager.py   # The app
└── run_experimental_app.bat      # Launcher
```

---

## 📚 Example Use Case: Turbine Blade Material Selection

**Goal:** Select best alloy for turbine blade operating at 600°C

**Step 1: Upload Data**
- Upload Inconel 718 yield strength curve
- Upload Inconel 625 yield strength curve
- Upload Rene 80 yield strength curve

**Step 2: Compare**
- Compare all three at 600°C
- Check which maintains highest strength
- Consider cost and availability

**Step 3: Decision**
- Rene 80: 850 MPa at 600°C 🏆
- Inconel 718: 650 MPa at 600°C
- Inconel 625: 550 MPa at 600°C

**Result:** Rene 80 selected for high-temperature application!

---

## 🎯 Next Steps

1. **Try the examples:**
   - Upload the 3 example CSV files
   - Visualize each curve
   - Compare them side-by-side

2. **Upload your own data:**
   - Gather experimental test results
   - Format as CSV
   - Upload and visualize

3. **Build your database:**
   - Add data from literature
   - Include lab test results
   - Track all your materials

---

## 🆘 Troubleshooting

### **CSV won't upload**
→ Check file format (must be .csv)
→ Ensure numeric data only
→ Remove any empty rows

### **Plot looks weird**
→ Check data units are consistent
→ Look for outliers or data entry errors
→ Verify temperature column is sorted

### **Can't find my data**
→ Check `experimental_data/` folder
→ Make sure you clicked "Save Data"
→ Refresh the page

---

## 🌟 Features Coming Soon

- Export plots as high-res images
- Curve fitting and interpolation
- Property prediction at temperatures
- Multi-property comparison
- Integration with ML models

---

**Start uploading your experimental data and visualize those temperature-dependent curves! 🌡️📊**
