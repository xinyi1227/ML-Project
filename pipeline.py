import os
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = 'MLEndHWII_sample_400'

def parse_filename(filename):
    # Format: S<ID>_<type>_<take>_<Song>.wav
    parts = filename.replace('.wav', '').split('_')
    if len(parts) >= 4:
        participant_id = parts[0]
        interpretation_type = parts[1]
        song_label = parts[-1]
        return participant_id, interpretation_type, song_label
    return None, None, None

def extract_features(file_path):
    try:
        # Load audio - prompt says 10 seconds duration input
        y, sr = librosa.load(file_path, sr=None, duration=10.0) 
        
        # Pad if less than 10 seconds (optional, but good for consistency if using spectrograms)
        # For mean/std MFCC stats, padding doesn't strictly matter as much as for CNNs, 
        # but let's just use what we have.
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        # Stats
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        return np.concatenate([mfcc_mean, mfcc_std])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.wav')]
    print(f"Found {len(files)} files.")
    
    X = []
    y = []
    groups = []
    
    # Process a subset for testing if needed, but 400 is small enough
    for i, f in enumerate(files):
        if i % 50 == 0:
            print(f"Processing file {i}/{len(files)}")
            
        pid, itype, label = parse_filename(f)
        if pid is None:
            continue
            
        feat = extract_features(os.path.join(DATA_DIR, f))
        if feat is not None:
            X.append(feat)
            y.append(label)
            groups.append(pid)
            
    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Classification - Use GroupKFold to ensure independence
    gkf = GroupKFold(n_splits=5)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print("Starting Cross-Validation...")
    y_pred = cross_val_predict(clf, X_scaled, y, groups=groups, cv=gkf)
    
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Also just unique labels to understand the task difficulty
    print("Unique labels:", np.unique(y))

if __name__ == "__main__":
    main()


