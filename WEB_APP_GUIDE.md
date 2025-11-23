# 🌐 Web Application Guide

## Launching the Web Dashboard

### Method 1: Double-Click (Easiest!) ⭐
1. Navigate to: `C:\Users\marcu\OneDrive\Desktop\Digital Foundry\ML-Alloy-Project\`
2. **Double-click** `run_webapp.bat`
3. Your web browser will automatically open the dashboard!

### Method 2: Command Line
1. Open Command Prompt
2. Run:
   ```
   cd "C:\Users\marcu\OneDrive\Desktop\Digital Foundry\ML-Alloy-Project"
   run_webapp.bat
   ```

### Method 3: Manual
1. Activate your environment:
   ```
   cd "C:\Users\marcu\OneDrive\Desktop\Digital Foundry\ML-Alloy-Project"
   activate.bat
   ```
2. Launch Streamlit:
   ```
   streamlit run app.py
   ```

---

## 🎯 What You'll See

The web dashboard will open in your browser at: `http://localhost:8501`

### Five Main Pages:

#### 🏠 **Home**
- Project overview
- Quick statistics
- Data summary
- Visual dashboard

#### 📥 **Download Data**
- Interactive form to query Materials Project
- Select alloy systems (Fe-Ni, Ti-Al, etc.)
- Filter by properties (metallic, stable, etc.)
- Download data with one click
- View instant previews

#### 📊 **Explore Data**
- Interactive visualizations
- Formation energy vs composition
- Density analysis
- Stability distributions
- Correlation heatmaps
- Crystal structure analysis
- Filter and sort data
- Download filtered datasets

#### 🤖 **Train Models**
- Build ML models visually
- Select features and target properties
- Compare multiple algorithms:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
- View performance metrics (R², MAE, RMSE)
- Interactive prediction plots
- No coding required!

#### 🔮 **Discover Alloys**
- (Coming soon!)
- Generate novel candidates
- Predict properties
- Rank by stability

---

## 💡 Key Features

### Interactive Charts
- **Hover** over points to see details
- **Zoom** by clicking and dragging
- **Pan** by holding shift and dragging
- **Download** charts as PNG

### Real-Time Updates
- Changes happen instantly
- No page refresh needed
- Live data filtering

### Beautiful Visualizations
- Plotly interactive charts
- Color-coded by properties
- Multiple chart types
- Customizable views

### No Coding Required!
- Point-and-click interface
- Form-based inputs
- Visual feedback
- User-friendly design

---

## 🚀 Quick Workflow

### 1. Start the App
Double-click `run_webapp.bat`

### 2. Download Data
- Go to "📥 Download Data"
- Select system (e.g., Ti-Al)
- Choose filters
- Click "Download Data"
- Wait for download to complete

### 3. Explore
- Go to "📊 Explore Data"
- Select your downloaded dataset
- View interactive charts
- Analyze correlations
- Check statistics

### 4. Train Models
- Go to "🤖 Train Models"
- Select features and target
- Click "Train Models"
- Compare algorithm performance
- View predictions

### 5. Download Results
- Export filtered data
- Save charts as images
- Document your findings

---

## 📊 Example Use Case

**Goal:** Find stable Ti-Al alloys for aerospace applications

1. **Download**: Get Ti-Al system data (metallic, all compositions)
2. **Explore**: Visualize formation energy vs composition
3. **Filter**: Show only materials with E_hull < 0.1 eV
4. **Identify**: Find compositions with high stability
5. **Train**: Build model to predict properties of new compositions
6. **Export**: Download promising candidates

**All in your web browser - no coding!**

---

## 🎨 Interface Tips

### Navigation
- Use sidebar to switch between pages
- Check "Project Info" for status

### Data Selection
- Dropdown menus for easy selection
- Checkboxes for filters
- Sliders for ranges

### Visualizations
- Tabs organize different views
- Click legends to hide/show series
- Download plots with camera icon

### Tables
- Sortable by clicking column headers
- Scrollable for large datasets
- Downloadable as CSV

---

## 🛑 Stopping the App

To stop the web server:
1. Go to the command prompt window
2. Press `Ctrl + C`
3. Confirm if asked

Or simply close the command prompt window.

---

## 🔧 Troubleshooting

### "streamlit: command not found"
→ Make sure you activated the environment first
→ Try running `run_webapp.bat` instead

### "Port 8501 is already in use"
→ Another instance is running
→ Close other terminal windows or use: `streamlit run app.py --server.port 8502`

### Browser doesn't open automatically
→ Manually go to: http://localhost:8501

### Charts not showing
→ Refresh the page
→ Check that data file exists

### Download button not working
→ Make sure you've selected a dataset
→ Check that data meets filter criteria

---

## 📱 Mobile Access

The dashboard works on mobile browsers too!
- Find your computer's IP address
- Access from phone: `http://[your-ip]:8501`

---

## 🎯 Next Steps

Once you're comfortable with the web app:
1. Download multiple alloy systems
2. Build comprehensive datasets
3. Train advanced ML models
4. Compare different material systems
5. Identify novel alloy candidates

---

## 🌟 Advantages Over Command Line

| Feature | Web App | Command Line |
|---------|---------|--------------|
| Ease of use | ✅ Point & click | ❌ Type commands |
| Visualizations | ✅ Interactive | ⚠️ Static images |
| Data exploration | ✅ Real-time filters | ❌ Manual filtering |
| Model comparison | ✅ Side-by-side | ❌ Separate runs |
| Sharing | ✅ Share URL | ❌ Share code |
| Learning curve | ✅ Minimal | ❌ Steeper |

---

## 📚 Resources

- **Streamlit Docs**: https://docs.streamlit.io/
- **Plotly Docs**: https://plotly.com/python/
- **Materials Project**: https://next-gen.materialsproject.org/

---

**Enjoy your interactive ML Alloy Development Dashboard! 🚀**
