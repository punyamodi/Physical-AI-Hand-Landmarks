import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def train_gesture_model(csv_path: str, model_name: str = "hand_gesture_model.pkl"):
    """
    Trains a simple Random Forest classifier on collected landmark data.
    This demonstrates the end-to-end Physical AI pipeline.
    """
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Feature columns: x0, y0, z0 ... x20, y20, z20
    # Ignoring timestamp, label (target), handedness, score
    feature_cols = [col for col in df.columns if col.startswith(('x', 'y', 'z')) and col[1:].isdigit()]
    
    X = df[feature_cols].values
    y = df['label'].values
    
    # Encode labels
    labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(labels)}
    y_encoded = np.array([label_map[label] for label in y])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} samples for {len(labels)} classes: {labels}")
    
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred, target_names=labels))
    
    joblib.dump({
        'model': clf,
        'label_map': {v: k for k, v in label_map.items()},
        'features': feature_cols
    }, model_name)
    
    print(f"\nModel saved as {model_name}")

if __name__ == "__main__":
    # Example usage
    data_dir = "data/"
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if csv_files:
        latest_csv = os.path.join(data_dir, csv_files[-1])
        train_gesture_model(latest_csv)
    else:
        print("No training data found in data/ folder. Collect some data first using ui.py!")
