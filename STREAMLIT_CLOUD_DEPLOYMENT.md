# 🚀 Streamlit Cloud Deployment Guide

## Complete Guide to Deploying Your Materials Science Toolkit

This guide will help you deploy all three Streamlit apps (Materials Project Explorer, GNN Predictor, and CALPHAD Tools) to Streamlit Cloud so you can share them with anyone via a public URL.

## 📋 Prerequisites

1. **GitHub Account** (free): https://github.com
2. **Streamlit Cloud Account** (free): https://streamlit.io/cloud
3. **Materials Project API Key** (optional, free): https://materialsproject.org/api

## 🎯 Step-by-Step Deployment

### Step 1: Prepare Your GitHub Repository

#### 1.1 Initialize Git Repository

```bash
cd "C:\Users\marcu\OneDrive\Desktop\Digital Foundry\ML-Alloy-Project"

# Initialize git
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit - Digital Foundry Materials Science Toolkit"
```

#### 1.2 Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `digital-foundry-materials` (or your choice)
3. Description: "Materials Science Toolkit with MP Explorer, GNN Predictor, and CALPHAD Tools"
4. **Public** (required for free Streamlit Cloud)
5. Click "Create repository"

#### 1.3 Push to GitHub

```bash
# Add remote origin (replace with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/digital-foundry-materials.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 2: Deploy to Streamlit Cloud

#### 2.1 Sign Up for Streamlit Cloud

1. Go to https://streamlit.io/cloud
2. Click "Sign up" and connect with GitHub
3. Authorize Streamlit to access your repositories

#### 2.2 Deploy Your App

1. Click "New app" button
2. Fill in the details:
   - **Repository**: Select `your-username/digital-foundry-materials`
   - **Branch**: `main`
   - **Main file path**: `Home.py`
   - **App URL**: Choose a custom URL (e.g., `digital-foundry-materials`)

3. Click "Deploy!"

#### 2.3 Wait for Deployment

- Streamlit Cloud will install dependencies from `requirements.txt`
- This may take 5-10 minutes for first deployment
- Watch the logs for any errors

### Step 3: Configure Secrets (Optional but Recommended)

If you have a Materials Project API key:

1. In Streamlit Cloud dashboard, go to your app
2. Click the three dots menu → "Settings"
3. Go to "Secrets" section
4. Add your secrets:

```toml
# Materials Project API Key
MP_API_KEY = "your_actual_api_key_here"
```

5. Click "Save"
6. App will automatically restart

### Step 4: Test Your Deployment

1. Open your app URL: `https://your-app-name.streamlit.app`
2. Test each page:
   - ✅ Home page loads
   - ✅ Materials Project Explorer works
   - ✅ GNN Property Predictor loads
   - ✅ CALPHAD Tools accessible
3. Test key features:
   - Upload CSV files
   - Generate phase diagrams
   - Create TDB files

## 📁 Project Structure (Final)

Your repository should have this structure:

```
digital-foundry-materials/
├── Home.py                              # Main entry point
├── pages/
│   ├── 1_Materials_Project_Explorer.py  # App 1
│   ├── 2_GNN_Property_Predictor.py      # App 2
│   └── 3_CALPHAD_Tools.py               # App 3
├── espei_integration.py                 # ESPEI module
├── requirements.txt                     # Dependencies
├── .streamlit/
│   ├── config.toml                      # Configuration
│   └── secrets.toml.example             # Secrets template
├── .gitignore                           # Git ignore rules
├── README.md                            # Project README
├── test_espei_data.csv                  # Sample data
└── docs/                                # Documentation
    ├── MULTI_PHASE_ESPEI_GUIDE.md
    ├── ESPEI_QUICK_START.md
    └── STREAMLIT_CLOUD_DEPLOYMENT.md
```

## ⚙️ Configuration Files

### requirements.txt
Already created with all necessary dependencies:
- streamlit
- pycalphad, espei
- pandas, numpy
- plotly
- scikit-learn
- And all dependencies

### .streamlit/config.toml
Streamlit configuration for theme and settings.

### .gitignore
Excludes large files, secrets, and temporary data from Git.

## 🔧 Troubleshooting

### Problem: Deployment Fails with "Module not found"

**Solution:**
```bash
# Make sure requirements.txt is complete
# Check the error log for missing packages
# Add missing packages to requirements.txt
git add requirements.txt
git commit -m "Update requirements"
git push
```

### Problem: Large Files Cause Push to Fail

**Solution:**
```bash
# Remove large files from git
git rm --cached large_file.csv
git rm --cached *.tdb

# Make sure .gitignore includes them
echo "*.tdb" >> .gitignore
echo "*.csv" >> .gitignore

git commit -m "Remove large files"
git push
```

### Problem: App Runs Locally but Not on Cloud

**Common causes:**
1. **Absolute file paths**: Use relative paths
   ```python
   # Bad
   file = "C:/Users/marcu/data.csv"

   # Good
   file = "data.csv"  # or Path("data.csv")
   ```

2. **Missing dependencies**: Check requirements.txt

3. **API keys**: Make sure secrets are configured in Streamlit Cloud

### Problem: "Out of Memory" Error

**Solutions:**
1. Reduce dataset size for examples
2. Use smaller models
3. Consider upgrading to Streamlit Cloud paid tier (more resources)

### Problem: TDB Files Not Found

**Solution:**
- TDB files generated by users are temporary on cloud
- Don't rely on persistent file storage
- Provide download buttons for users to save locally
- Consider using Streamlit session state for temporary storage

## 🎨 Customization

### Custom Domain

Free tier uses: `your-app-name.streamlit.app`

For custom domain (requires paid plan):
1. Go to App Settings
2. Add custom domain
3. Configure DNS records

### Custom Theme

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF6B6B"        # Main accent color
backgroundColor = "#FFFFFF"      # Background
secondaryBackgroundColor = "#F0F2F6"  # Sidebar
textColor = "#262730"           # Text
font = "sans serif"             # Font family
```

### Analytics

Add Google Analytics (requires custom HTML):
1. Create `components.html` with GA code
2. Use `st.components.v1.html()` to include

## 📊 Resource Limits (Free Tier)

- **RAM**: 1 GB
- **CPU**: Shared
- **Storage**: Ephemeral (files don't persist between sessions)
- **Bandwidth**: Unlimited
- **Apps**: Unlimited public apps
- **Sleep**: Apps sleep after inactivity (wake on visit)

**Tips for staying within limits:**
- Keep example datasets small
- Don't store large TDB files in repo
- Use efficient data structures
- Cache heavy computations with `@st.cache_data`

## 🔐 Security Best Practices

### Never Commit Secrets

```bash
# Make sure .gitignore includes:
.streamlit/secrets.toml
.env
*.key
```

### Use Streamlit Secrets

Instead of hardcoding:
```python
# Bad
API_KEY = "my_secret_key"

# Good
import streamlit as st
API_KEY = st.secrets.get("MP_API_KEY", None)
```

### Validate User Inputs

```python
# Always validate file uploads
if uploaded_file:
    if uploaded_file.size > 10_000_000:  # 10 MB limit
        st.error("File too large")
    if not uploaded_file.name.endswith('.csv'):
        st.error("Only CSV files allowed")
```

## 🔄 Updating Your Deployed App

### Make Changes Locally

```bash
# Edit your files
# Test locally
streamlit run Home.py
```

### Push Updates

```bash
git add .
git commit -m "Add new feature"
git push
```

### Automatic Redeployment

- Streamlit Cloud automatically detects changes
- Rebuilds and redeploys automatically
- Usually takes 2-3 minutes

### Manual Reboot

If needed:
1. Go to app dashboard
2. Click three dots menu
3. Select "Reboot app"

## 📱 Sharing Your App

### Get Your URL

After deployment: `https://your-app-name.streamlit.app`

### Share With Others

1. **Direct Link**: Just share the URL!
2. **QR Code**: Generate QR code for the URL
3. **Embed**: Use iframe to embed in website:
   ```html
   <iframe src="https://your-app-name.streamlit.app"
           width="100%" height="800px"></iframe>
   ```

### Make README Prominent

Add badges to README.md:
```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
```

## 📈 Monitoring

### View Logs

1. Go to app dashboard
2. Click "Logs" tab
3. See real-time application logs
4. Debug errors

### Usage Analytics

Streamlit Cloud provides:
- Visitor count
- Session duration
- Error rates
- Resource usage

Access via app dashboard → "Analytics"

## 🚀 Advanced: Multiple Deployments

### Development vs Production

Deploy two versions:

**Development:**
- Branch: `dev`
- URL: `your-app-dev.streamlit.app`

**Production:**
- Branch: `main`
- URL: `your-app.streamlit.app`

### Workflow

```bash
# Develop in dev branch
git checkout -b dev
# Make changes, test

# When ready for production
git checkout main
git merge dev
git push
```

## 💡 Best Practices

1. **Keep dependencies minimal**: Only include what you need
2. **Use caching**: `@st.cache_data` for expensive operations
3. **Handle errors gracefully**: Try/except blocks with user-friendly messages
4. **Provide examples**: Include sample data in repo
5. **Document well**: Clear instructions in app and README
6. **Test locally first**: Always test before pushing
7. **Monitor regularly**: Check logs for errors
8. **Update dependencies**: Keep packages current for security

## 📞 Getting Help

### Streamlit Community

- **Forum**: https://discuss.streamlit.io
- **Discord**: https://discord.gg/streamlit
- **GitHub**: https://github.com/streamlit/streamlit

### Common Issues Database

- Check Streamlit docs: https://docs.streamlit.io
- Search community forum
- Review GitHub issues

## ✅ Deployment Checklist

Before going live:

- [ ] All files committed to Git
- [ ] Pushed to GitHub public repository
- [ ] requirements.txt is complete and tested
- [ ] .gitignore excludes large/secret files
- [ ] README.md is informative
- [ ] Secrets configured (if needed)
- [ ] App deployed to Streamlit Cloud
- [ ] All three pages load correctly
- [ ] Key features tested
- [ ] Error handling works
- [ ] Documentation is clear
- [ ] URL is shareable

## 🎉 Success!

Once deployed, your app is:
- ✅ Publicly accessible
- ✅ Automatically updated on git push
- ✅ Free to use (within limits)
- ✅ Shareable via URL
- ✅ Mobile-friendly
- ✅ Professionally hosted

**Your Materials Science Toolkit is now live for the world to use!** 🚀

---

## Quick Reference Commands

```bash
# Initialize repo
git init
git add .
git commit -m "Initial commit"

# Create remote and push
git remote add origin https://github.com/USERNAME/REPO.git
git push -u origin main

# Update app
git add .
git commit -m "Update message"
git push

# Check status
git status
git log --oneline

# Create dev branch
git checkout -b dev
git push -u origin dev
```

**Ready to deploy? Follow the steps above and your app will be live in minutes!** 🎊
