import pandas as pd
import glob
import os
import subprocess
import re

# Config
LITERATURE_DIR = "../literatura"
BENCHMARK_FILE = "../BrenchmarkCORPUSMEJORADO.ods"
OUTPUT_FILE = "analysis_advanced.txt"

def analyze_benchmark():
    print("Analyzing Benchmark ODS...")
    try:
        df = pd.read_excel(BENCHMARK_FILE, engine='odf')
        # Filter for relevant keywords in "Modelo(s)" or "Variables"
        # Looking for Feature Weighting or Contexts
        print("Checking for Hybrid/Weighting approaches in ODS...")
        cols_to_check = [c for c in df.columns if 'Modelo' in c or 'Variables' in c]
        for idx, row in df.iterrows():
            content = " ".join([str(row[c]) for c in cols_to_check])
            if re.search(r'weight|hybrid|context|híbrido|ponderaci', content, re.IGNORECASE):
                print(f"Match in ODS Row {idx}: {row['ID Artículo (Nombre Archivo)']}")
        return df
    except Exception as e:
        print(f"Error reading ODS: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    try:
        result = subprocess.run(['pdftotext', '-layout', pdf_path, '-'], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
             return ""
        return result.stdout
    except Exception:
        return ""

def search_advanced_keywords(text, filename):
    results = {}
    
    # 1. Feature Weighting / Hybridization
    if re.search(r'feature weighting|weighted feature|hybrid|hibrid|context|weight', text, re.IGNORECASE):
        # Look for specific context
        m = re.search(r'([^.]*?(?:feature weighting|hybrid approach|weighted)[^.]*?\.)', text, re.IGNORECASE)
        if m: results['Hybridization_Context'] = m.group(1).strip()

    # 2. Specific Authors
    authors = {
        'Saidani': r'Saidani',
        'Mwita': r'Mwita', 
        'Salas-Velasco': r'Salas.?Velasco', 
        'Ayaneh': r'Ayaneh'
    }
    
    for auth, pattern in authors.items():
        if re.search(pattern, text, re.IGNORECASE):
            # Extract citation context
            m = re.search(r'([^.]*?' + pattern + r'[^.]*?\.)', text, re.IGNORECASE)
            if m: results[f'Citation_{auth}'] = m.group(1).strip()

    # 3. XGBoost vs Cox & Proportional Hazards
    if re.search(r'proportional hazard|PH assumption|violation|cox', text, re.IGNORECASE):
        if re.search(r'violation|fail|not hold|non-linear', text, re.IGNORECASE):
             m = re.search(r'([^.]*?(?:proportional hazard|PH assumption|Cox)[^.]*?(?:violat|fail|hold)[^.]*?\.)', text, re.IGNORECASE)
             if m: results['Cox_Violation'] = m.group(1).strip()

    # 4. Feature Selection (LASSO / RFE)
    if re.search(r'LASSO|RFE|Recursive Feature Elimination|Small Data', text, re.IGNORECASE):
         m = re.search(r'([^.]*?(?:LASSO|RFE|Recursive Feature Elimination)[^.]*?\.)', text, re.IGNORECASE)
         if m: results['Feature_Selection'] = m.group(1).strip()

    return results

def main():
    analyze_benchmark()
    
    pdf_files = glob.glob(os.path.join(LITERATURE_DIR, "*.pdf"))
    print(f"\nScanning {len(pdf_files)} PDFs for Advanced Topics...")
    
    for pdf in pdf_files:
        filename = os.path.basename(pdf)
        text = extract_text_from_pdf(pdf)
        if not text: continue
        
        res = search_advanced_keywords(text, filename)
        if res:
            print(f"--- {filename} ---")
            for k, v in res.items():
                print(f"[{k}]: {v}")
            print("")

if __name__ == "__main__":
    main()
