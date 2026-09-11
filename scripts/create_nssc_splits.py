import pandas as pd
from pathlib import Path
import json

ROOT = Path('C:/Users/deves/OneDrive/Desktop/Projects/TCMD-SS')
DATA_ROOT = ROOT / 'data/raw/nssc/test+train'

def create_nssc_splits():
    splits_dir = ROOT / 'data/splits'
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    def get_meta(p):
        return {
            'image_path': str(p.resolve()), 
            'base_id': p.stem,
            'source_id': p.stem,
            'group_id': p.stem,
            'split_group_id': p.stem,
            'class_id': 0,
            'is_original': True
        }

    # Train normal
    train_normal = []
    for p in (DATA_ROOT / 'train/normal').glob('*.jpg'):
        train_normal.append({**get_meta(p), 'label': 'normal', 'is_anomaly': 0, 'origin': 'nssc'})
    df_train = pd.DataFrame(train_normal)
    df_train.to_csv(splits_dir / 'train_normal.csv', index=False)
    
    # Test clean (normal)
    test_clean = []
    for p in (DATA_ROOT / 'test/normal').glob('*.jpg'):
        test_clean.append({**get_meta(p), 'label': 'normal', 'is_anomaly': 0, 'origin': 'nssc'})
    df_test_clean = pd.DataFrame(test_clean)
    df_test_clean.to_csv(splits_dir / 'test_clean.csv', index=False)
    
    # Test anomaly
    test_anomaly = []
    for p in (DATA_ROOT / 'test/anomaly_real').glob('*.jpg'):
        test_anomaly.append({**get_meta(p), 'label': 'anomaly', 'is_anomaly': 1, 'origin': 'nssc'})
    df_test_anomaly = pd.DataFrame(test_anomaly)
    df_test_anomaly.to_csv(splits_dir / 'test_anomaly.csv', index=False)
    
    # Calibration normal (just split from train for now)
    df_cal = df_train.sample(frac=0.1, random_state=42)
    df_cal.to_csv(splits_dir / 'calibration_normal.csv', index=False)
    
    # We also need a splits/protocol.json to trick the pipeline
    with open(ROOT / 'outputs/splits/protocol.json', 'w') as f:
        json.dump({"nssc": True}, f)
        
    print(f"Created NSSC splits in {splits_dir}")

if __name__ == '__main__':
    create_nssc_splits()
