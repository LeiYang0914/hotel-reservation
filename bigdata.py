from pathlib import Path
import csv
import math
import os
import re
import sys

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


RAW_DATA_PATH = Path("Hotel Reservations.csv")
CLEAN_DATA_PATH = Path("hotel_reservations_clean.csv")
TARGET_COLUMN = "booking_status"
LABEL_COLUMN = "booking_status_label"
RANDOM_SEED = 42


EXPECTED_COLUMNS = [
    "Booking_ID",
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "type_of_meal_plan",
    "required_car_parking_space",
    "room_type_reserved",
    "lead_time",
    "arrival_year",
    "arrival_month",
    "arrival_date",
    "market_segment_type",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",
    "no_of_special_requests",
    "booking_status",
]

INTEGER_COLUMNS = [
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "required_car_parking_space",
    "lead_time",
    "arrival_year",
    "arrival_month",
    "arrival_date",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "no_of_special_requests",
]

NUMERIC_COLUMNS = [*INTEGER_COLUMNS, "avg_price_per_room"]

CATEGORICAL_COLUMNS = [
    "type_of_meal_plan",
    "room_type_reserved",
    "market_segment_type",
    "booking_status",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "type_of_meal_plan",
    "room_type_reserved",
    "market_segment_type",
]

BINARY_COLUMNS = [
    "required_car_parking_space",
    "repeated_guest",
]

VALID_CATEGORY_VALUES = {
    "type_of_meal_plan": ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"],
    "room_type_reserved": [
        "Room_Type 1",
        "Room_Type 2",
        "Room_Type 3",
        "Room_Type 4",
        "Room_Type 5",
        "Room_Type 6",
        "Room_Type 7",
    ],
    "market_segment_type": [
        "Aviation",
        "Complementary",
        "Corporate",
        "Offline",
        "Online",
    ],
    "booking_status": ["Canceled", "Not_Canceled"],
}

ONE_HOT_BASELINES = {
    "type_of_meal_plan": "Meal Plan 1",
    "room_type_reserved": "Room_Type 1",
    "market_segment_type": "Online",
}

SCALABLE_FEATURE_COLUMNS = [
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "lead_time",
    "arrival_year",
    "arrival_month",
    "arrival_date",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",
    "no_of_special_requests",
    "total_nights",
    "total_guests",
    "arrival_day_of_week",
    "arrival_quarter",
    "estimated_booking_value",
    "avg_price_per_guest",
    "previous_total_bookings",
    "previous_cancellation_rate",
]

BINARY_FEATURE_COLUMNS = [
    "required_car_parking_space",
    "repeated_guest",
    "has_children",
    "arrival_is_weekend",
]


def configure_local_spark_environment() -> None:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    if os.environ.get("JAVA_HOME"):
        return

    project_root = Path(__file__).resolve().parent
    java_home_candidates = [
        java_path.parent.parent
        for java_path in (project_root / ".jdk").glob("*/bin/java.exe")
    ]
    java_home_candidates.extend(
        [
            java_path.parent.parent
            for root in [Path("C:/Program Files/Java"), Path("C:/Program Files/Eclipse Adoptium")]
            if root.exists()
            for java_path in root.glob("*/bin/java.exe")
        ]
    )
    java_home_candidates.extend(
        [
        java_path.parent.parent
        for root in [Path("C:/Program Files/JetBrains"), Path("C:/Program Files/Java")]
        if root.exists()
        for java_path in root.glob("*/jbr/bin/java.exe")
        ]
    )

    if java_home_candidates:
        os.environ["JAVA_HOME"] = str(java_home_candidates[0])


def create_spark_session() -> SparkSession:
    configure_local_spark_environment()

    return (
        SparkSession.builder.appName("HotelReservationDataCleaning")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.ansi.enabled", "false")
        .getOrCreate()
    )


def load_dataset(spark: SparkSession, path: Path = RAW_DATA_PATH) -> DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    schema = T.StructType([T.StructField(column, T.StringType(), True) for column in EXPECTED_COLUMNS])
    return (
        spark.read.option("header", True)
        .option("mode", "PERMISSIVE")
        .schema(schema)
        .csv(str(path))
    )


def validate_schema(df: DataFrame) -> None:
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    extra_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))

    if missing_columns or extra_columns:
        raise ValueError(
            "Dataset schema mismatch. "
            f"Missing columns: {missing_columns}. Extra columns: {extra_columns}."
        )


def standardize_text_columns(df: DataFrame) -> DataFrame:
    clean = df
    for column in EXPECTED_COLUMNS:
        clean = clean.withColumn(column, F.trim(F.col(column)))

    return clean


def coerce_numeric_columns(df: DataFrame) -> DataFrame:
    clean = df
    for column in INTEGER_COLUMNS:
        clean = clean.withColumn(column, F.col(column).cast(T.IntegerType()))

    clean = clean.withColumn("avg_price_per_room", F.col("avg_price_per_room").cast(T.DoubleType()))
    return clean


def add_quality_check_columns(df: DataFrame) -> DataFrame:
    date_string = F.concat_ws(
        "-",
        F.col("arrival_year").cast("string"),
        F.lpad(F.col("arrival_month").cast("string"), 2, "0"),
        F.lpad(F.col("arrival_date").cast("string"), 2, "0"),
    )

    clean = (
        df.withColumn("arrival_date_full", F.to_date(date_string, "yyyy-MM-dd"))
        .withColumn("total_nights_check", F.col("no_of_weekend_nights") + F.col("no_of_week_nights"))
        .withColumn("total_guests_check", F.col("no_of_adults") + F.col("no_of_children"))
    )

    missing_required_condition = None
    for column in EXPECTED_COLUMNS:
        condition = F.col(column).isNull()
        missing_required_condition = condition if missing_required_condition is None else missing_required_condition | condition

    negative_numeric_condition = None
    for column in NUMERIC_COLUMNS:
        condition = F.col(column) < 0
        negative_numeric_condition = condition if negative_numeric_condition is None else negative_numeric_condition | condition

    invalid_category_condition = None
    for column, valid_values in VALID_CATEGORY_VALUES.items():
        condition = F.col(column).isNull() | ~F.col(column).isin(valid_values)
        invalid_category_condition = condition if invalid_category_condition is None else invalid_category_condition | condition

    invalid_binary_condition = None
    for column in BINARY_COLUMNS:
        condition = F.col(column).isNull() | ~F.col(column).isin([0, 1])
        invalid_binary_condition = condition if invalid_binary_condition is None else invalid_binary_condition | condition

    return (
        clean.withColumn("missing_required_value", F.coalesce(missing_required_condition, F.lit(False)))
        .withColumn("invalid_arrival_date", F.col("arrival_date_full").isNull())
        .withColumn("zero_or_negative_night_stay", F.col("total_nights_check") <= 0)
        .withColumn("zero_or_negative_guest_count", F.col("total_guests_check") <= 0)
        .withColumn("negative_numeric_value", F.coalesce(negative_numeric_condition, F.lit(False)))
        .withColumn("invalid_binary_value", F.coalesce(invalid_binary_condition, F.lit(False)))
        .withColumn("invalid_category_value", F.coalesce(invalid_category_condition, F.lit(False)))
        .withColumn(
            "invalid_target_value",
            F.col(TARGET_COLUMN).isNull() | ~F.col(TARGET_COLUMN).isin(VALID_CATEGORY_VALUES[TARGET_COLUMN]),
        )
    )


def invalid_record_condition() -> Column:
    return (
        F.col("missing_required_value")
        | F.col("invalid_arrival_date")
        | F.col("zero_or_negative_night_stay")
        | F.col("zero_or_negative_guest_count")
        | F.col("negative_numeric_value")
        | F.col("invalid_binary_value")
        | F.col("invalid_category_value")
        | F.col("invalid_target_value")
    )


def analyze_dataset(df: DataFrame, title: str) -> None:
    row_count = df.count()
    print(f"\n{title}")
    print(f"Dataset shape: {row_count:,} rows x {len(df.columns)} columns")
    print(f"Target column: {TARGET_COLUMN}")

    print("\nColumn audit:")
    df.printSchema()

    print("\nMissing values:")
    missing_exprs = [
        F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(column)
        for column in df.columns
    ]
    df.select(missing_exprs).show(truncate=False)

    print("\nTarget distribution:")
    (
        df.groupBy(TARGET_COLUMN)
        .agg(F.count("*").alias("count"))
        .withColumn("percent", F.round(F.col("count") / F.lit(row_count) * 100, 2))
        .orderBy(F.desc("count"))
        .show(truncate=False)
    )

    print("\nCategorical distributions:")
    for column in CATEGORICAL_COLUMNS:
        print(f"\n{column}")
        df.groupBy(column).count().orderBy(F.desc("count")).show(truncate=False)

    print("\nNumeric summary:")
    df.select(NUMERIC_COLUMNS).summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max").show(truncate=False)


def print_quality_findings(df: DataFrame) -> None:
    checked = add_quality_check_columns(df)
    duplicate_rows = df.count() - df.distinct().count()
    duplicate_booking_ids = df.count() - df.select("Booking_ID").distinct().count()

    print("\nData quality findings:")
    print("Duplicate full rows:", duplicate_rows)
    print("Duplicate Booking_ID values:", duplicate_booking_ids)
    checked.select(
        F.sum(F.col("invalid_arrival_date").cast("int")).alias("invalid_arrival_dates"),
        F.sum(F.col("zero_or_negative_night_stay").cast("int")).alias("zero_or_negative_night_stays"),
        F.sum(F.col("zero_or_negative_guest_count").cast("int")).alias("zero_or_negative_guest_count"),
        F.sum((F.col("avg_price_per_room") <= 0).cast("int")).alias("non_positive_room_price"),
    ).show(truncate=False)

    print("Unknown category labels:")
    for column, valid_values in VALID_CATEGORY_VALUES.items():
        unknown = (
            df.select(column)
            .where(~F.col(column).isin(valid_values))
            .distinct()
            .orderBy(column)
            .collect()
        )
        print(f"{column}: {[row[column] for row in unknown]}")


def preprocess_dataset(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    validate_schema(df)
    clean = standardize_text_columns(df)
    clean = coerce_numeric_columns(clean)
    clean = add_quality_check_columns(clean)

    rejected = clean.where(invalid_record_condition()).select(
        "Booking_ID",
        F.col("missing_required_value").cast("int"),
        F.col("invalid_arrival_date").cast("int"),
        F.col("zero_or_negative_night_stay").cast("int"),
        F.col("zero_or_negative_guest_count").cast("int"),
        F.col("negative_numeric_value").cast("int"),
        F.col("invalid_binary_value").cast("int"),
        F.col("invalid_category_value").cast("int"),
        F.col("invalid_target_value").cast("int"),
    )

    final_columns = [
        "Booking_ID",
        "no_of_adults",
        "no_of_children",
        "no_of_weekend_nights",
        "no_of_week_nights",
        "type_of_meal_plan",
        "required_car_parking_space",
        "room_type_reserved",
        "lead_time",
        "arrival_year",
        "arrival_month",
        "arrival_date",
        "arrival_date_full",
        "market_segment_type",
        "repeated_guest",
        "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled",
        "avg_price_per_room",
        "no_of_special_requests",
        "booking_status",
    ]

    clean = (
        clean.where(~invalid_record_condition())
        .select(final_columns)
        .orderBy("arrival_date_full", "Booking_ID")
    )
    return clean, rejected


def validate_clean_dataset(clean: DataFrame) -> None:
    missing_count = clean.select(
        [
            F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(column)
            for column in clean.columns
        ]
    ).collect()[0].asDict()

    missing_columns = {column: count for column, count in missing_count.items() if count}
    if missing_columns:
        raise ValueError(f"Clean dataset still has missing values: {missing_columns}")

    duplicate_booking_ids = clean.count() - clean.select("Booking_ID").distinct().count()
    if duplicate_booking_ids:
        raise ValueError(f"Clean dataset contains {duplicate_booking_ids} duplicate Booking_ID values.")

    quality = add_quality_check_columns(clean.drop("arrival_date_full"))
    invalid_count = quality.where(invalid_record_condition()).count()
    if invalid_count:
        raise ValueError(f"Clean dataset still has {invalid_count} invalid rows.")


def sanitize_column_suffix(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", value)).strip("_")


def add_engineered_features(clean: DataFrame) -> DataFrame:
    previous_total_bookings = (
        F.col("no_of_previous_cancellations")
        + F.col("no_of_previous_bookings_not_canceled")
    )

    return (
        clean.withColumn("total_nights", F.col("no_of_weekend_nights") + F.col("no_of_week_nights"))
        .withColumn("total_guests", F.col("no_of_adults") + F.col("no_of_children"))
        .withColumn("has_children", (F.col("no_of_children") > 0).cast("int"))
        .withColumn("arrival_day_of_week", F.dayofweek(F.col("arrival_date_full")))
        .withColumn("arrival_is_weekend", F.col("arrival_day_of_week").isin([1, 7]).cast("int"))
        .withColumn("arrival_quarter", F.quarter(F.col("arrival_date_full")))
        .withColumn(
            "estimated_booking_value",
            F.round(F.col("avg_price_per_room") * F.col("total_nights"), 2),
        )
        .withColumn(
            "avg_price_per_guest",
            F.round(F.col("avg_price_per_room") / F.col("total_guests"), 2),
        )
        .withColumn("previous_total_bookings", previous_total_bookings)
        .withColumn(
            "previous_cancellation_rate",
            F.when(previous_total_bookings > 0, F.col("no_of_previous_cancellations") / previous_total_bookings)
            .otherwise(F.lit(0.0)),
        )
        .withColumn(LABEL_COLUMN, F.when(F.col(TARGET_COLUMN) == "Canceled", 1).otherwise(0))
    )


def add_one_hot_features(df: DataFrame) -> tuple[DataFrame, list[str]]:
    encoded = df
    encoded_columns = []

    for column in CATEGORICAL_FEATURE_COLUMNS:
        baseline = ONE_HOT_BASELINES[column]
        for value in VALID_CATEGORY_VALUES[column]:
            if value == baseline:
                continue

            encoded_column = f"{column}_{sanitize_column_suffix(value)}"
            encoded = encoded.withColumn(encoded_column, (F.col(column) == value).cast("int"))
            encoded_columns.append(encoded_column)

    return encoded, encoded_columns


def build_feature_dataset(clean: DataFrame) -> tuple[DataFrame, list[str], list[str]]:
    engineered = add_engineered_features(clean)
    encoded, encoded_columns = add_one_hot_features(engineered)

    feature_columns = SCALABLE_FEATURE_COLUMNS + BINARY_FEATURE_COLUMNS + encoded_columns
    selected_columns = ["Booking_ID"] + feature_columns + [LABEL_COLUMN]

    return encoded.select(selected_columns), feature_columns, encoded_columns


def print_correlation_analysis(feature_df: DataFrame, feature_columns: list[str]) -> None:
    correlation_columns = feature_columns + [LABEL_COLUMN]
    vector_df = VectorAssembler(
        inputCols=correlation_columns,
        outputCol="correlation_features",
        handleInvalid="skip",
    ).transform(feature_df.select(correlation_columns))

    matrix = Correlation.corr(vector_df, "correlation_features", "pearson").head()[0]
    label_index = len(correlation_columns) - 1

    target_correlations = []
    for index, column in enumerate(feature_columns):
        value = float(matrix[index, label_index])
        if not math.isnan(value):
            target_correlations.append((column, value, abs(value)))

    target_correlations.sort(key=lambda item: item[2], reverse=True)

    print("\nCORRELATION CHECK")
    print("Top features correlated with booking cancellation label:")
    for column, correlation, _ in target_correlations[:12]:
        print(f"{column}: {correlation:.4f}")

    high_feature_pairs = []
    for left_index, left_column in enumerate(feature_columns):
        for right_index in range(left_index + 1, len(feature_columns)):
            right_column = feature_columns[right_index]
            value = float(matrix[left_index, right_index])
            if not math.isnan(value):
                high_feature_pairs.append((left_column, right_column, value, abs(value)))

    high_feature_pairs.sort(key=lambda item: item[3], reverse=True)

    print("\nHighest feature-to-feature correlations:")
    for left_column, right_column, correlation, _ in high_feature_pairs[:10]:
        print(f"{left_column} vs {right_column}: {correlation:.4f}")


def split_train_test(feature_df: DataFrame) -> tuple[DataFrame, DataFrame]:
    split_df = feature_df.withColumn(
        "_split_bucket",
        F.pmod(F.xxhash64(F.col("Booking_ID")), F.lit(100)),
    )
    train_df = split_df.where(F.col("_split_bucket") < 80).drop("_split_bucket")
    test_df = split_df.where(F.col("_split_bucket") >= 80).drop("_split_bucket")
    return train_df, test_df


def calculate_train_scaling_stats(train_df: DataFrame) -> dict[str, tuple[float, float]]:
    aggregate_expressions = []
    for column in SCALABLE_FEATURE_COLUMNS:
        aggregate_expressions.append(F.mean(F.col(column)).alias(f"{column}_mean"))
        aggregate_expressions.append(F.stddev_samp(F.col(column)).alias(f"{column}_std"))

    stats = train_df.select(aggregate_expressions).collect()[0].asDict()
    return {
        column: (
            float(stats[f"{column}_mean"]),
            float(stats[f"{column}_std"] or 0.0),
        )
        for column in SCALABLE_FEATURE_COLUMNS
    }


def apply_standardization(
    df: DataFrame,
    scaling_stats: dict[str, tuple[float, float]],
    encoded_columns: list[str],
) -> DataFrame:
    normalized = df
    scaled_columns = []

    for column in SCALABLE_FEATURE_COLUMNS:
        mean_value, std_value = scaling_stats[column]
        scaled_column = f"{column}_scaled"
        if std_value == 0:
            normalized = normalized.withColumn(scaled_column, F.lit(0.0))
        else:
            normalized = normalized.withColumn(
                scaled_column,
                F.round((F.col(column) - F.lit(mean_value)) / F.lit(std_value), 6),
            )
        scaled_columns.append(scaled_column)

    final_columns = scaled_columns + BINARY_FEATURE_COLUMNS + encoded_columns + [LABEL_COLUMN]
    return normalized.select(final_columns)


def prepare_train_test_datasets(
    feature_df: DataFrame,
    encoded_columns: list[str],
) -> tuple[DataFrame, DataFrame]:
    train_df, test_df = split_train_test(feature_df)
    scaling_stats = calculate_train_scaling_stats(train_df)
    train_normalized = apply_standardization(train_df, scaling_stats, encoded_columns)
    test_normalized = apply_standardization(test_df, scaling_stats, encoded_columns)

    return train_normalized, test_normalized


def write_single_csv(df: DataFrame, output_path: Path) -> int:
    if output_path.exists():
        output_path.unlink()

    row_count = 0
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(df.columns)
        for row in df.toLocalIterator():
            writer.writerow([row[column] for column in df.columns])
            row_count += 1

    return row_count


def main() -> None:
    spark = create_spark_session()
    try:
        raw = load_dataset(spark)
        validate_schema(raw)

        raw_for_analysis = coerce_numeric_columns(standardize_text_columns(raw))
        analyze_dataset(raw_for_analysis, "DATA UNDERSTANDING")
        print_quality_findings(raw_for_analysis)

        clean, rejected = preprocess_dataset(raw)
        validate_clean_dataset(clean)

        feature_df, feature_columns, encoded_columns = build_feature_dataset(clean)
        print_correlation_analysis(feature_df, feature_columns)

        train_normalized, test_normalized = prepare_train_test_datasets(feature_df, encoded_columns)

        clean_row_count = write_single_csv(clean, CLEAN_DATA_PATH)
        train_row_count = train_normalized.count()
        test_row_count = test_normalized.count()
        rejected_row_count = rejected.count()

        print("\nPREPROCESSING COMPLETE")
        print(f"Rows retained: {clean_row_count:,} / {clean_row_count + rejected_row_count:,}")
        print(f"Rows removed: {rejected_row_count:,}")
        print(f"Clean dataset saved to: {CLEAN_DATA_PATH}")

        print("\nClean dataset quality checks:")
        print("Missing values: 0")
        print("Duplicate Booking_ID values: 0")
        print("Invalid quality-rule rows: 0")
        print("Target distribution:")
        clean.groupBy(TARGET_COLUMN).count().orderBy(F.desc("count")).show(truncate=False)

        print("\nTrain/test split:")
        print(f"Train rows: {train_row_count:,}")
        print(f"Test rows: {test_row_count:,}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
