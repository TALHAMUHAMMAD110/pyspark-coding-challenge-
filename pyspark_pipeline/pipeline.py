import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, BooleanType, TimestampType

# Define the schemas for the raw input dataframes as specified in the document.

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
    StructField("config_id", IntegerType(), True),  # Corresponds to item_id
    StructField("simple_id", IntegerType(), True),
    StructField("occurred_at", StringType(), True)
])

previous_orders_schema = StructType([
    StructField("order_date", StringType(), True), # The document says 'date' type, but it's typically read as a string in Spark
    StructField("customer_id", IntegerType(), True),
    StructField("config_id", IntegerType(), True),  # Corresponds to item_id
    StructField("simple_id", IntegerType(), True),
    StructField("occurred_at", StringType(), True)
])

# Define a single function that encapsulates the entire pipeline.
# This makes it easy to test and integrate into a larger system.
def create_training_inputs(impressions_df, clicks_df, add_to_carts_df, previous_orders_df):
    """
    This function processes raw dataframes to create the training input data set.
    
    Args:
        impressions_df: A Spark DataFrame containing impressions data.
        clicks_df: A Spark DataFrame containing click data.
        add_to_carts_df: A Spark DataFrame containing add-to-cart data.
        previous_orders_df: A Spark DataFrame containing previous order data.
    
    Returns:
        A Spark DataFrame with the final training inputs.
        The DataFrame will have these columns:
        - impressions (int)
        - actions (array<int>, length=1000, padded with 0)
        - action_types (array<int>, length=1000, padded with 0)
        - is_order (boolean)
    """
    
    # -------------------------------------------------------------------------
    # Step 1: Standardize and combine all action dataframes
    # This prepares all customer actions into a single, unified format.
    # The `action_type` column is crucial for the final output.
    # We also rename columns to a common standard for easier union.
    # -------------------------------------------------------------------------
    
    # Process clicks
    clicks_with_type = clicks_df.withColumn("action_type", F.lit(1)) \
        .withColumn("action_timestamp", F.col("click_time")) \
        .withColumn("item_id", F.col("item_id")) \
        .select("customer_id", "item_id", "action_timestamp", "action_type")
        
    # Process add-to-carts
    add_to_carts_with_type = add_to_carts_df.withColumn("action_type", F.lit(2)) \
        .withColumn("action_timestamp", F.col("occurred_at")) \
        .withColumn("item_id", F.col("config_id")) \
        .select("customer_id", "item_id", "action_timestamp", "action_type")

    # Process previous orders
    # Note: The document specifies 'order_date' but we need a timestamp for sorting
    # We will assume 'occurred_at' is the correct timestamp field for sorting
    previous_orders_with_type = previous_orders_df.withColumn("action_type", F.lit(3)) \
        .withColumn("action_timestamp", F.col("occurred_at")) \
        .withColumn("item_id", F.col("config_id")) \
        .select("customer_id", "item_id", "action_timestamp", "action_type")

    # Combine all actions into a single DataFrame
    all_actions_df = clicks_with_type.union(add_to_carts_with_type).union(previous_orders_with_type)
    all_actions_df = all_actions_df.withColumn("action_timestamp", F.to_timestamp(F.col("action_timestamp")))
    all_actions_df = all_actions_df.withColumn("action_date", F.to_date("action_timestamp"))

    # -------------------------------------------------------------------------
    # Step 2: Process impressions
    # -------------------------------------------------------------------------
    exploded_impressions_df = impressions_df.withColumn("impression", F.explode("impressions")) \
        .withColumn("impression_item_id", F.col("impression.item_id")) \
        .withColumn("is_order", F.col("impression.is_order")) \
        .withColumn("impression_date", F.to_date("dt"))
    
    # -------------------------------------------------------------------------
    # Step 3: Filter valid actions only before impression_dat within 1 year lookback
    # -------------------------------------------------------------------------
    valid_actions_df = all_actions_df.join(
        exploded_impressions_df.select("customer_id", "impression_date").distinct(),
        on="customer_id",
        how="inner"
    ).filter(
        (F.col("action_date") < F.col("impression_date")) &
        (F.col("action_date") >= F.date_sub(F.col("impression_date"), 365))
    )

    # -------------------------------------------------------------------------
    # Step 4: Rank and collect top 1000
    # -------------------------------------------------------------------------
    window_spec = Window.partitionBy("customer_id", "impression_date") \
        .orderBy(F.col("action_timestamp").desc())

    ranked_actions_df = valid_actions_df.withColumn("rn", F.row_number().over(window_spec))
    top_1000_actions = ranked_actions_df.filter(F.col("rn") <= 1000)

    # Note: The use of `groupBy` and `agg` with `collect_list` can be
    # memory-intensive. For production, consider an alternative if this becomes
    # a bottleneck. A common approach is a second window function with `collect_list`
    # and then filtering for the first row of each customer, but the current
    # approach is more readable for a smaller dataset.
    customer_actions_df = top_1000_actions.groupBy("customer_id", "impression_date").agg(
        F.collect_list("item_id").alias("actions"),
        F.collect_list("action_type").alias("action_types")
    )
    
    # -------------------------------------------------------------------------
    # Step 5: Join back with impressions
    # -------------------------------------------------------------------------
    final_df = exploded_impressions_df.join(
        F.broadcast(customer_actions_df),
        on=["customer_id", "impression_date"],
        how="left"
    )

    # -------------------------------------------------------------------------
    # Step 6: Pad to length 1000 and select final columns
    # -------------------------------------------------------------------------
    # Pad the lists to a length of 1000 with a value of 0.
    num_actions_col = F.size("actions")
    padded_actions = F.when(F.col("actions").isNotNull() & (num_actions_col < 1000), 
                            F.concat("actions", F.array_repeat(F.lit(0), 1000 - num_actions_col))) \
                      .when(F.col("actions").isNotNull(), F.col("actions")) \
                      .otherwise(F.array_repeat(F.lit(0), 1000))
    
    padded_action_types = F.when(F.col("action_types").isNotNull() & (num_actions_col < 1000), 
                                 F.concat("action_types", F.array_repeat(F.lit(0), 1000 - num_actions_col))) \
                           .when(F.col("action_types").isNotNull(), F.col("action_types")) \
                           .otherwise(F.array_repeat(F.lit(0), 1000))
    
    # Select the final columns as specified by the document, including `is_order`
    training_data = final_df.select(
        F.col("impression_item_id").alias("impressions"),
        padded_actions.alias("actions"),
        padded_action_types.alias("action_types"),
        F.col("is_order")
    )

    return training_data

