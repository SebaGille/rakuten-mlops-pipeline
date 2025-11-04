"""Utility script to fix corrupted inference log CSV"""
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

def fix_inference_log():
    """Fix corrupted inference log by cleaning and standardizing format"""
    
    inference_log_path = PROJECT_ROOT / "data" / "monitoring" / "inference_log.csv"
    backup_path = PROJECT_ROOT / "data" / "monitoring" / "inference_log_backup.csv"
    
    if not inference_log_path.exists():
        print("No inference log found. Nothing to fix.")
        return
    
    print(f"Reading log file: {inference_log_path}")
    
    try:
        # Create backup
        import shutil
        shutil.copy(inference_log_path, backup_path)
        print(f"✓ Backup created: {backup_path}")
        
        # Read CSV with error handling
        df = pd.read_csv(inference_log_path, on_bad_lines='skip', encoding='utf-8')
        print(f"✓ Read {len(df)} valid rows")
        
        # Handle old column names
        column_mapping = {
            'predicted_prdtypecode': 'predicted_class',
            'text_len': 'designation_length'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
                print(f"✓ Renamed column '{old_col}' to '{new_col}'")
        
        # Ensure required columns exist
        required_columns = {
            'timestamp': None,
            'designation': '',
            'description': '',
            'predicted_class': None,
            'designation_length': 0,
            'description_length': 0
        }
        
        for col, default_value in required_columns.items():
            if col not in df.columns:
                df[col] = default_value
                print(f"✓ Added missing column '{col}'")
        
        # Clean timestamp format
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            print(f"✓ Cleaned timestamps, {len(df)} rows remaining")
        
        # Calculate missing lengths if needed
        if 'designation_length' in df.columns and df['designation_length'].isna().any():
            df.loc[df['designation_length'].isna(), 'designation_length'] = df.loc[
                df['designation_length'].isna(), 'designation'
            ].str.len()
        
        if 'description_length' in df.columns and df['description_length'].isna().any():
            df.loc[df['description_length'].isna(), 'description_length'] = df.loc[
                df['description_length'].isna(), 'description'
            ].fillna('').str.len()
        
        # Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates(subset=['timestamp', 'designation'], keep='first')
        if len(df) < initial_len:
            print(f"✓ Removed {initial_len - len(df)} duplicate rows")
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Select final columns in proper order
        final_columns = [
            'timestamp', 'designation', 'description', 'predicted_class',
            'designation_length', 'description_length'
        ]
        
        df = df[final_columns]
        
        # Write cleaned CSV
        df.to_csv(inference_log_path, index=False)
        print(f"✓ Cleaned log written: {len(df)} rows")
        print(f"✓ Columns: {list(df.columns)}")
        print("\n✅ Inference log fixed successfully!")
        print(f"   Original backup saved at: {backup_path}")
        
    except Exception as e:
        print(f"❌ Error fixing log: {e}")
        if backup_path.exists():
            print(f"   Original backup available at: {backup_path}")
        return

if __name__ == "__main__":
    fix_inference_log()


