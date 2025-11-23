# How to Activate Your Environment

## ⚠️ Important: Don't Double-Click!
Double-clicking `activate.bat` will flash a window and close immediately.
You need to run it from a terminal/command prompt.

---

## Method 1: Command Prompt (CMD) - EASIEST

1. **Press `Windows Key + R`**
2. **Type:** `cmd`
3. **Press Enter**
4. **Copy and paste this command:**
   ```
   cd "C:\Users\marcu\OneDrive\Desktop\Digital Foundry\ML-Alloy-Project" && activate.bat
   ```
5. **Press Enter**

You should see:
```
========================================
  ML Alloy Development Project
========================================

Activating virtual environment...

Environment activated!
Python:
Python 3.13.9

To deactivate, type: deactivate
========================================
(mp-alloy-env) C:\Users\marcu\OneDrive\Desktop\Digital Foundry\ML-Alloy-Project>
```

**Note:** The `(mp-alloy-env)` at the start means it's working!

---

## Method 2: PowerShell

1. **Press `Windows Key + X`**
2. **Select "Windows PowerShell" or "Terminal"**
3. **Copy and paste this command:**
   ```powershell
   cd "C:\Users\marcu\OneDrive\Desktop\Digital Foundry\ML-Alloy-Project"
   .\mp-alloy-env\Scripts\Activate.ps1
   ```
4. **Press Enter**

**If you get an error about execution policies:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then try the activation command again.

---

## Method 3: Direct Activation (Manual)

1. **Open Command Prompt or PowerShell**
2. **Navigate to project:**
   ```
   cd "C:\Users\marcu\OneDrive\Desktop\Digital Foundry\ML-Alloy-Project"
   ```
3. **Activate directly:**

   **For Command Prompt:**
   ```
   mp-alloy-env\Scripts\activate.bat
   ```

   **For PowerShell:**
   ```powershell
   .\mp-alloy-env\Scripts\Activate.ps1
   ```

---

## How to Know It's Working

When activated, you'll see:
- `(mp-alloy-env)` at the start of your command prompt
- You can run: `python --version` and see `Python 3.13.9`
- You can run: `python -c "import mp_api; print('OK')"` and see `OK`

---

## Quick Test After Activation

Once activated, test everything works:
```bash
python test_mp_connection.py
```

You should see:
```
SUCCESS! Connected to Materials Project
Material ID: mp-13
Formula: Fe
```

---

## Deactivating

To deactivate when you're done:
```
deactivate
```

The `(mp-alloy-env)` will disappear from your prompt.

---

## Troubleshooting

### "The system cannot find the path specified"
→ Make sure you're in the right directory first:
```
cd "C:\Users\marcu\OneDrive\Desktop\Digital Foundry\ML-Alloy-Project"
```

### "activate.bat is not recognized"
→ Make sure you're in the project directory (see above)

### PowerShell: "execution of scripts is disabled"
→ Run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Still not working?
→ Use the direct method:
```
cd "C:\Users\marcu\OneDrive\Desktop\Digital Foundry\ML-Alloy-Project"
mp-alloy-env\Scripts\activate.bat
```

---

## Once Activated, You Can:

```bash
# Test connection
python test_mp_connection.py

# Download data
python download_first_dataset.py

# Start Python
python

# Run any Python script
python your_script.py
```

---

**Remember:** You need to activate every time you open a new terminal/command prompt!
