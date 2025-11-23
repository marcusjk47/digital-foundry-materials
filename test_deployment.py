"""
Test script to verify deployment setup before pushing to Streamlit Cloud
"""

import sys
from pathlib import Path
import importlib.util

def test_file_structure():
    """Check that all required files exist"""
    print("Checking file structure...")

    required_files = [
        "Home.py",
        "pages/1_Materials_Project_Explorer.py",
        "pages/2_GNN_Property_Predictor.py",
        "pages/3_CALPHAD_Tools.py",
        "requirements.txt",
        ".streamlit/config.toml",
        ".gitignore",
        "README.md",
        "espei_integration.py",
    ]

    all_good = True
    for file in required_files:
        filepath = Path(file)
        if filepath.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING!")
            all_good = False

    return all_good

def test_imports():
    """Check that all required packages can be imported"""
    print("\nChecking package imports...")

    required_packages = [
        "streamlit",
        "pandas",
        "numpy",
        "plotly",
        "pycalphad",
        "espei",
        "sklearn",
        "requests",
    ]

    all_good = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED!")
            all_good = False

    return all_good

def test_requirements():
    """Check that requirements.txt is valid"""
    print("\nChecking requirements.txt...")

    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("  ✗ requirements.txt not found!")
        return False

    with open(req_file, 'r') as f:
        lines = f.readlines()

    print(f"  ✓ Found {len(lines)} package specifications")

    # Check for key packages
    content = ''.join(lines)
    key_packages = ["streamlit", "pycalphad", "espei", "pandas"]

    for pkg in key_packages:
        if pkg in content:
            print(f"  ✓ {pkg} specified")
        else:
            print(f"  ✗ {pkg} NOT in requirements.txt!")
            return False

    return True

def test_page_structure():
    """Check that page files are Python files"""
    print("\nChecking page files...")

    pages = [
        "pages/1_Materials_Project_Explorer.py",
        "pages/2_GNN_Property_Predictor.py",
        "pages/3_CALPHAD_Tools.py",
    ]

    all_good = True
    for page in pages:
        filepath = Path(page)
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                if 'streamlit' in content:
                    print(f"  ✓ {page} contains Streamlit code")
                else:
                    print(f"  ⚠ {page} may not be a Streamlit app")
            except Exception as e:
                print(f"  ✗ {page} - Error reading: {e}")
                all_good = False
        else:
            print(f"  ✗ {page} - NOT FOUND!")
            all_good = False

    return all_good

def test_no_absolute_paths():
    """Check for hardcoded absolute paths"""
    print("\nChecking for absolute paths...")

    files_to_check = [
        "Home.py",
        "pages/1_Materials_Project_Explorer.py",
        "pages/2_GNN_Property_Predictor.py",
        "pages/3_CALPHAD_Tools.py",
        "espei_integration.py",
    ]

    problematic_patterns = [
        "C:/Users/",
        "C:\\Users\\",
        "/Users/marcu",
    ]

    issues = []
    for file in files_to_check:
        filepath = Path(file)
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                for pattern in problematic_patterns:
                    if pattern in content:
                        issues.append(f"{file}: Contains '{pattern}'")
            except:
                pass

    if issues:
        print("  ⚠ Found absolute paths (may cause issues on cloud):")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print("  ✓ No absolute paths detected")
        return True

def test_gitignore():
    """Check that .gitignore excludes important files"""
    print("\nChecking .gitignore...")

    gitignore = Path(".gitignore")
    if not gitignore.exists():
        print("  ✗ .gitignore not found!")
        return False

    with open(gitignore, 'r') as f:
        content = f.read()

    important_patterns = [
        "*.tdb",
        "*.csv",
        "secrets.toml",
        "__pycache__",
        ".env",
    ]

    all_good = True
    for pattern in important_patterns:
        if pattern in content:
            print(f"  ✓ Excludes {pattern}")
        else:
            print(f"  ⚠ Should exclude {pattern}")
            all_good = False

    return all_good

def main():
    """Run all tests"""
    print("=" * 60)
    print("STREAMLIT CLOUD DEPLOYMENT - PRE-FLIGHT CHECK")
    print("=" * 60)
    print()

    tests = [
        ("File Structure", test_file_structure),
        ("Package Imports", test_imports),
        ("Requirements File", test_requirements),
        ("Page Structure", test_page_structure),
        ("Absolute Paths", test_no_absolute_paths),
        ("Git Ignore", test_gitignore),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nError running {test_name}: {e}")
            results.append((test_name, False))
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:>8} - {test_name}")

    all_passed = all(result for _, result in results)

    print()
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print()
        print("Your app is ready for deployment to Streamlit Cloud!")
        print()
        print("Next steps:")
        print("1. git init")
        print("2. git add .")
        print("3. git commit -m 'Initial commit'")
        print("4. Create GitHub repo and push")
        print("5. Deploy on streamlit.io/cloud")
        print()
        print("See STREAMLIT_CLOUD_DEPLOYMENT.md for details")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print()
        print("Please fix the issues above before deploying.")
        print("See error messages for details.")

    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
