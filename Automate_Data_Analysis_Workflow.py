import pandas as pd
import numpy as np
import sqlite3
import json
import logging
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configure Logging
logging.basicConfig(filename='pipeline.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def fetch_db_data(query, db_path):
    """Fetch data from SQLite database"""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(query, conn)
    conn.close()
    logging.info("Data fetched successfully.")
    return df

def clean_data(df):
    """Clean data by handling missing values and duplicates"""
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    logging.info("Data cleaned successfully.")
    return df

def create_features(df):
    """Generate new features"""
    df['log_value'] = df['value'].apply(lambda x: np.log1p(x))
    df['is_weekend'] = df['timestamp'].dt.weekday >= 5
    logging.info("Feature engineering completed.")
    return df

def train_model(df):
    """Train a RandomForest ML model"""
    X = df[['log_value', 'is_weekend']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model trained with accuracy: {accuracy:.2f}")
    
    return model, accuracy

def save_results(model, accuracy, filename='model_results.json'):
    """Save model results to a JSON file"""
    results = {'model_type': str(type(model)), 'params': model.get_params(), 'accuracy': accuracy}
    with open(filename, 'w') as f:
        json.dump(results, f)
    logging.info("Results saved successfully.")

def main():
    """Main function to execute the pipeline"""
    query = "SELECT * FROM dataset"
    db_path = "database.db"
    
    df = fetch_db_data(query, db_path)
    df = clean_data(df)
    df = create_features(df)
    model, accuracy = train_model(df)
    save_results(model, accuracy)

if __name__ == "__main__":
    main()
