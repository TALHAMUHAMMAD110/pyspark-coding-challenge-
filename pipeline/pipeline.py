import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType,
    IntegerType, ArrayType, BooleanType
)

# =========================
# Schema Definitions
# =========================

impressions_schema = StructType([
    StructField("dt", StringType(), True),
    StructField("ranking_id", StringType(), True),
    StructField("customer_id", IntegerType(), True),
    StructField("impressions", ArrayType(
        StructType([
            StructField("item_id", IntegerType(), True),
            StructField("is_order", BooleanType(), True)
        ])
    ), True)
])

clicks_schema = StructType([
    StructField("dt", StringType(), True),
    StructField("customer_id", IntegerType(), True),
    StructField("item_id", IntegerType(), True),
    StructField("click_time", StringType(), True)
])

add_to_carts_schema = StructType([
    StructField("dt", StringType(), True),
    StructField("customer_id", IntegerType(), True),
    StructField("config_id", IntegerType(), True),
    StructField("simple_id", IntegerType(), True),
    StructField("occurred_at", StringType(), True)
])

previous_orders_schema = StructType([
    StructField("order_date", StringType(), True),
    StructField("customer_id", IntegerType(), True),
    StructField("config_id", IntegerType(), True),
    StructField("simple_id", IntegerType(), True),
    StructField("occurred_at", StringType(), True)
])


# =========================
# Pipeline Function
# =========================

def create_training_inputs(impressions_df, clicks_df, add_to_carts_df, previous_orders_df):
    """
    Build training dataset for transformer model.

    Args:
        impressions_df: impressions DataFrame
        clicks_df: clicks DataFrame
        add_to_carts_df: add-to-carts DataFrame
        previous_orders_df: previous orders DataFrame

    Returns:
        Spark DataFrame with columns:
            - impressions (int)
            - actions (array<int>, length=1000, padded with 0)
            - action_types (array<int>, length=1000, padded with 0)
    """

    # =========================
    # Step 1: Standardize and unify all actions
    # =========================
    clicks_with_type = clicks_df.withColumn("action_type", F.lit(1)) \
        .withColumn("action_timestamp", F.col("click_time")) \
        .withColumn("item_id", F.col("item_id")) \
        .select("customer_id", "item_id", "action_timestamp", "action_type")

    add_to_carts_with_type = add_to_carts_df.withColumn("action_type", F.lit(2)) \
        .withColumn("action_timestamp", F.col("occurred_at")) \
        .withColumn("item_id", F.col("config_id")) \
        .select("customer_id", "item_id", "action_timestamp", "action_type")

    previous_orders_with_type = previous_orders_df.withColumn("action_type", F.lit(3)) \
        .withColumn("action_timestamp", F.col("occurred_at")) \
        .withColumn("item_id", F.col("config_id")) \
        .select("customer_id", "item_id", "action_timestamp", "action_type")

    all_actions_df = clicks_with_type.union(add_to_carts_with_type).union(previous_orders_with_type)
    all_actions_df = all_actions_df.withColumn("action_date", F.to_date("action_timestamp"))

    # =========================
    # Step 2: Process impressions
    # =========================
    exploded_impressions_df = impressions_df.withColumn("impression", F.explode("impressions")) \
        .withColumn("impression_item_id", F.col("impression.item_id")) \
        .withColumn("impression_date", F.to_date("dt"))

    # =========================
    # Step 3: Filter valid actions
    # - only before impression_date
    # - within 1 year lookback
    # =========================
    valid_actions_df = all_actions_df.join(
        exploded_impressions_df.select("customer_id", "impression_date").distinct(),
        on="customer_id",
        how="inner"
    ).filter(
        (F.col("action_date") < F.col("impression_date")) &
        (F.col("action_date") >= F.date_sub(F.col("impression_date"), 365))
    )

    # =========================
    # Step 4: Rank and collect top 1000
    # =========================
    window_spec = Window.partitionBy("customer_id", "impression_date") \
        .orderBy(F.col("action_timestamp").desc())

    ranked_actions_df = valid_actions_df.withColumn("rn", F.row_number().over(window_spec))
    top_1000_actions = ranked_actions_df.filter(F.col("rn") <= 1000)

    customer_actions_df = top_1000_actions.groupBy("customer_id", "impression_date").agg(
        F.collect_list("item_id").alias("actions"),
        F.collect_list("action_type").alias("action_types")
    )

    # =========================
    # Step 5: Join back with impressions
    # =========================
    final_df = exploded_impressions_df.join(
        customer_actions_df,
        on=["customer_id", "impression_date"],
        how="left"
    )

    # =========================
    # Step 6: Pad to length 1000
    # =========================
    num_actions_col = F.size("actions")

    padded_actions = F.when(num_actions_col < 1000,
                            F.concat("actions", F.array_repeat(F.lit(0), 1000 - num_actions_col))) \
        .otherwise(F.col("actions"))

    padded_action_types = F.when(num_actions_col < 1000,
                                F.concat("action_types", F.array_repeat(F.lit(0), 1000 - num_actions_col))) \
        .otherwise(F.col("action_types"))

    return final_df.select(
        F.col("impression_item_id").alias("impressions"),
        padded_actions.alias("actions"),
        padded_action_types.alias("action_types")
    )
