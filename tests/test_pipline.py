import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType


from pyspark_pipeline.pipeline import create_training_inputs, impressions_schema, clicks_schema, add_to_carts_schema, previous_orders_schema

# Define a pytest fixture to create a SparkSession
@pytest.fixture(scope="session")
def spark_session():
    """
    Creates a temporary SparkSession for testing purposes.
    """
    spark = SparkSession.builder \
        .appName("PySpark Unit Tests") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    yield spark
    spark.stop()

# --- Unit Tests ---
def test_create_training_inputs_returns_correct_schema(spark_session):
    """
    Test that the create_training_inputs function returns a DataFrame
    with the expected schema and column names.
    """
    # Create some dummy data for the test
    impressions_data = [("2023-01-01", "id1", 1, [(101, True), (102, False)])]
    clicks_data = [("2022-12-30", 1, 103, "2022-12-30 10:00:00")]
    add_to_carts_data = [("2022-12-29", 1, 105, 1, "2022-12-29 09:00:00")]
    previous_orders_data = [("2022-12-28", 1, 106, 1, "2022-12-28 08:00:00")]

    # Create dummy DataFrames
    impressions_df = spark_session.createDataFrame(impressions_data, impressions_schema)
    clicks_df = spark_session.createDataFrame(clicks_data, clicks_schema)
    add_to_carts_df = spark_session.createDataFrame(add_to_carts_data, add_to_carts_schema)
    previous_orders_df = spark_session.createDataFrame(previous_orders_data, previous_orders_schema)

    # Call the pipeline function
    result_df = create_training_inputs(impressions_df, clicks_df, add_to_carts_df, previous_orders_df)

    # Define the expected schema for the output
    expected_schema = StructType([
        StructField("impressions", IntegerType(), True),
        StructField("actions", ArrayType(IntegerType(), True), True),
        StructField("action_types", ArrayType(IntegerType(), True), True)
    ])

    # Assert that the result's schema matches the expected schema
    assert result_df.schema == expected_schema

def test_create_training_inputs_correct_action_sequence_and_padding(spark_session):
    """
    Test that the pipeline correctly generates the action sequence and pads it
    to a length of 1000.
    """
    # Create data with a single impression and a few actions for one customer
    impressions_data = [("2023-01-01", "id1", 1, [(101, True)])]
    clicks_data = [("2022-12-31", 1, 104, "2022-12-31 11:00:00")] # Most recent
    add_to_carts_data = [("2022-12-29", 1, 105, 1, "2022-12-29 09:00:00")] # Oldest
    previous_orders_data = [("2022-12-30", 1, 106, 1, "2022-12-30 08:00:00")] # Middle

    # Create dummy DataFrames
    impressions_df = spark_session.createDataFrame(impressions_data, impressions_schema)
    clicks_df = spark_session.createDataFrame(clicks_data, clicks_schema)
    add_to_carts_df = spark_session.createDataFrame(add_to_carts_data, add_to_carts_schema)
    previous_orders_df = spark_session.createDataFrame(previous_orders_data, previous_orders_schema)

    # Call the pipeline function
    result_df = create_training_inputs(impressions_df, clicks_df, add_to_carts_df, previous_orders_df)
    
    result_row = result_df.first()

    # The expected actions and action types should be sorted in descending order of time
    expected_actions = [104, 106, 105] + [0] * 997 # 104 is most recent click, 106 is order, 105 is add-to-cart
    expected_action_types = [1, 3, 2] + [0] * 997  # Clicks=1, Orders=3, Add-to-Carts=2

    assert result_row["impressions"] == 101
    assert result_row["actions"] == expected_actions
    assert result_row["action_types"] == expected_action_types
    assert len(result_row["actions"]) == 1000
    assert len(result_row["action_types"]) == 1000
