"""
Quick training script to create a baseline XGBoost model
"""
import pandas as pd
import sys
import os
import joblib
from sklearn.model_selection import train_test_split

# Add src to path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, 'src'))

from labelers import CandlestickLabeler
from features import FeatureEngineer, create_targets
from models import ModelFactory
from utils import preprocess_ohlcv

# Find available data files
data_dir = os.path.join(base_dir, 'data', 'raw')
data_files = [f for f in os.listdir(data_dir) if f.endswith('_1h.parquet') or f.endswith('_1d.parquet')]

if not data_files:
    print("No data files found. Please run data_loader.py first.")
    sys.exit(1)

print(f"Found {len(data_files)} data files")

# Combine data from multiple stocks for better training
all_data = []
for file in data_files[:5]:  # Use first 5 files to speed up training
    print(f"Processing {file}...")
    df = pd.read_parquet(os.path.join(data_dir, file))
    df = preprocess_ohlcv(df)
    
    # Label patterns
    labeler = CandlestickLabeler(df)
    df = (labeler
        .label_doji()
        .label_hammer()
        .label_shooting_star()
        .label_engulfing()
        .label_harami()
        .get_labeled_data()
    )
    
    # Engineer features
    engineer = FeatureEngineer(df)
    df = (engineer
        .add_candle_features()
        .add_technical_indicators()
        .add_price_context()
        .add_volatility_features()
        .get_features()
    )
    
    # Create targets
    df = create_targets(df, horizon=1)
    
    all_data.append(df)

# Combine all data
print("Combining data...")
combined_df = pd.concat(all_data, ignore_index=False)
combined_df = combined_df.dropna()

print(f"Total samples: {len(combined_df)}")

# Prepare features and target
feature_cols = [col for col in combined_df.columns 
               if col not in ['target_return', 'target_direction', 'target_binary', 
                              'target_next_close', 'symbol', 'timestamp', 'datetime']]

X = combined_df[feature_cols]
y = combined_df['target_binary']

print(f"Features: {len(feature_cols)}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("\nTraining XGBoost model...")
model = ModelFactory.get_xgboost_model({
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1
})

model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"\nTrain Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")

# Save model
models_dir = os.path.join(base_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, 'xgboost_baseline.joblib')
joblib.dump(model, model_path)

# Save feature columns for later use
feature_cols_path = os.path.join(models_dir, 'feature_columns.joblib')
joblib.dump(feature_cols, feature_cols_path)

print(f"\nModel saved to {model_path}")
print(f"Feature columns saved to {feature_cols_path}")
print("\nTraining complete!")
