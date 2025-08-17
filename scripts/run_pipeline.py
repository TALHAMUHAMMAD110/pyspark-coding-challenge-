from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark_pipeline.pipeline import create_training_inputs,impressions_schema, clicks_schema, add_to_carts_schema, previous_orders_schema


if __name__ == "__main__":
    spark = SparkSession.builder.appName("SparkCodingChallenge").getOrCreate()

    # Example data
    impressions_data = [("2023-01-01", "id1", 1, [(101, True), (102, False)])]
    clicks_data = [("2022-12-30", 1, 103, "2022-12-30 10:00:00")]
    add_to_carts_data = [("2022-12-29", 1, 105, 1, "2022-12-29 09:00:00")]
    previous_orders_data = [("2022-12-28", 1, 106, 1, "2022-12-28 08:00:00")]

    impressions_df = spark.createDataFrame(impressions_data, impressions_schema)
    clicks_df = spark.createDataFrame(clicks_data, clicks_schema).withColumn("click_time", F.to_timestamp("click_time"))
    add_to_carts_df = spark.createDataFrame(add_to_carts_data, add_to_carts_schema)
    previous_orders_df = spark.createDataFrame(previous_orders_data, previous_orders_schema)

    training_df = create_training_inputs(impressions_df, clicks_df, add_to_carts_df, previous_orders_df)
    training_df.show(truncate=False)

    spark.stop()
