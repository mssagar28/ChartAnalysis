import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class ModelFactory:
    @staticmethod
    def get_xgboost_model(params=None):
        """Return configured XGBoost classifier"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'random_state': 42,
            'n_jobs': -1
        }
        if params:
            default_params.update(params)
        
        return xgb.XGBClassifier(**default_params)
    
    @staticmethod
    def get_lstm_model(input_shape, units=64, dropout=0.2, learning_rate=0.001):
        """Return compiled LSTM model"""
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout),
            BatchNormalization(),
            LSTM(units // 2, return_sequences=False),
            Dropout(dropout),
            BatchNormalization(),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

class TimeSeriesTrainer:
    def __init__(self, model, n_splits=5):
        self.model = model
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        
    def train_evaluate_xgboost(self, X, y):
        """Train and evaluate XGBoost with TimeSeriesSplit"""
        scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        
        for train_index, val_index in self.tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
            
            scores['accuracy'].append(accuracy_score(y_val, y_pred))
            scores['precision'].append(precision_score(y_val, y_pred))
            scores['recall'].append(recall_score(y_val, y_pred))
            scores['f1'].append(f1_score(y_val, y_pred))
            
        return pd.DataFrame(scores).mean()

    def train_lstm(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train LSTM model"""
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history
