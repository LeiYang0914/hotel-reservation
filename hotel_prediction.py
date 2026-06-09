import sys
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

try:
    # Import helper methods from bigdata
    from bigdata import (
        preprocess_dataset, 
        build_feature_dataset, 
        prepare_train_test_datasets
    )
    print("Successfully imported data-engineering methods from bigdata.py!")
except ImportError:
    print("\n[ERROR] Cannot find bigdata.py.")
    print("Please make sure bigdata.py is copied into your ~/hotel-reservation/ directory.")
    sys.exit(1)

# 1. Initialize Spark Session configured to run on Hadoop YARN
spark = SparkSession.builder \
    .appName("Hotel-Cancellation-Prediction") \
    .getOrCreate()

# Suppress verbose tracking logs
spark.sparkContext.setLogLevel("ERROR")

print("\n========== PHASE 1: LOADING RAW DATA FROM HADOOP HDFS ==========")
# Define the HDFS URL path pointing to the raw dataset location inside Hadoop
HDFS_RAW_PATH = "hdfs://localhost:9000/user/hotel_prediction/data/hotel_reservations.csv"

print(f"Reading raw data blocks from: {HDFS_RAW_PATH}")
raw_df = spark.read.csv(HDFS_RAW_PATH, header=True, inferSchema=True)
print(f"Raw rows ingested from HDFS: {raw_df.count():,}")

print("\n========== PHASE 2: RUNNING CUSTOM PREPROCESSING METHODS ==========")
print("Executing clean rules (Filtering invalid rows, dates, and handling text standardizations)...")
# Call custom cleaning code logic directly
clean_df, rejected_df = preprocess_dataset(raw_df)
print(f"-> Retained clean rows: {clean_df.count():,}")
print(f"-> Rejected/Dropped anomaly rows: {rejected_df.count():,}")

print("\nAssembling features and computing mathematical MinMaxScaler normalizations...")
# Call custom feature assembly and scaling code logic
feature_df, feature_cols, encoded_cols = build_feature_dataset(clean_df)

print("Splitting datasets into 80% Train and 20% Testing blocks...")
# Call custom train/test splitter method
train_normalized, test_normalized = prepare_train_test_datasets(feature_df, encoded_cols)

mllib_feature_cols = [c for c in train_normalized.columns if c != 'booking_status_label']
assembler = VectorAssembler(inputCols=mllib_feature_cols, outputCol='features')

train_assembled = assembler.transform(train_normalized)
test_assembled = assembler.transform(test_normalized)

# Cache data frames in worker node RAM to speed up subsequent iterative model training
train_assembled.cache()
test_assembled.cache()

print("\n========== PHASE 3: DISTRIBUTED DUAL MODEL TRAINING ==========")

# --- Model A: Random Forest ---
print("[1/2] Training Distributed Random Forest Classifier across YARN containers...")
rf = RandomForestClassifier(
    featuresCol="features", # Matches the final output vector column name from bigdata.py
    labelCol="booking_status_label", # Matches label mapping column name from bigdata.py
    numTrees=100,
    maxDepth=10,
    seed=42
)
rf_model = rf.fit(train_assembled)

# --- Model B: Logistic Regression ---
print("[2/2] Training Distributed Logistic Regression Classifier across YARN containers...")
lr = LogisticRegression(
    featuresCol="features", 
    labelCol="booking_status_label", 
    maxIter=20, 
    regParam=0.05, 
    elasticNetParam=0.1
)
lr_model = lr.fit(train_assembled)

print("\n========== PHASE 4: SIDE-BY-SIDE EVALUATION ==========")
# Generate classification test predictions
rf_predictions = rf_model.transform(test_assembled)
lr_predictions = lr_model.transform(test_assembled)

# Create evaluator measuring Area Under ROC
evaluator = BinaryClassificationEvaluator(
    labelCol="booking_status_label", 
    rawPredictionCol="rawPrediction", 
    metricName="areaUnderROC"
)

rf_auc = evaluator.evaluate(rf_predictions)
lr_auc = evaluator.evaluate(lr_predictions)

# Output summary metric presentation box to terminal console
print("\n" + "="*60)
print(f"{'Hadoop MLlib Model Evaluated':<38} | {'ROC-AUC Metric':<15}")
print("="*60)
print(f"{'Random Forest Classifier':<38} | {rf_auc:.4f}")
print(f"{'Logistic Regression Classifier':<38} | {lr_auc:.4f}")
print("="*60)

print("\n========== PHASE 5: SERIALIZING WINNING MODEL BINARY ==========")
model_save_base_path = "hdfs://localhost:9000/user/hotel_prediction/saved_models/best_bigdata_hadoop_model"

# Compare operational scores and persist the best variant state
if rf_auc >= lr_auc:
    print(f"Random Forest outperforms Logistic Regression! Saving model state...")
    rf_model.write().overwrite().save(model_save_base_path)
else:
    print(f"Logistic Regression outperforms Random Forest! Saving model state...")
    lr_model.write().overwrite().save(model_save_base_path)

print(f"Winning estimator model binary saved locally to: {model_save_base_path}")

spark.stop()