import sys
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel, LogisticRegressionModel

# 1. Initialize a local Spark Session to read and print model configurations
spark = SparkSession.builder \
    .appName("HDFS-Model-Viewer") \
    .master("local[*]") \
    .getOrCreate()

# Suppress overly verbose system logging
spark.sparkContext.setLogLevel("ERROR")

# Define the absolute HDFS path where the model is stored
HDFS_MODEL_PATH = "/user/hotel_prediction/saved_models/best_bigdata_hadoop_model"

print("\n" + "="*60)
print(f"LOADING SAVED BINARY MODEL FROM HDFS:\n{HDFS_MODEL_PATH}")
print("="*60)

# try-except sequence to load the correct structure dynamically.
try:
    print("[INFO] Attempting to load as a Random Forest Model...")
    model = RandomForestClassificationModel.load(HDFS_MODEL_PATH)
    
    print("\nSUCCESS: Loaded Random Forest Classifier.")
    print("\n--- TRAINED RANDOM FOREST DECISION NODES (First 2000 Chars) ---")
    print(model.toDebugString[:2000])

except Exception as e:
    # If it fails to load as a Random Forest, it means the LR model won
    print("[INFO] Not a Random Forest model. Attempting to load as Logistic Regression...")
    try:
        model = LogisticRegressionModel.load(HDFS_MODEL_PATH)
        
        print("\nSUCCESS: Loaded Logistic Regression Model.")
        print("\n--- LOGISTIC REGRESSION EQUATION METRICS ---")
        print(f"Intercept (Bias Element): {model.intercept}")
        print(f"Total Coefficients Optimized: {len(model.coefficients)}")
        print("\nCoefficients Vector Weights Array:")
        print(model.coefficients)
        
    except Exception as e_inner:
        print(f"\n[ERROR] Failed to load model from path. Ensure the path exists in HDFS.")
        print(f"Details: {e_inner}")
        spark.stop()
        sys.exit(1)

print("="*60)
print("Model payload extraction complete.")
spark.stop()