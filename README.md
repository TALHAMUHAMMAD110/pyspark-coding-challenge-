# PySpark Coding Challenge

## Overview
This project implements a PySpark pipeline that prepares **training input data** for a transformer model.  
The input data includes:
- Impressions (items shown to a customer in the "Our Choices for You" carousel)
- Customer actions (clicks, add-to-carts, and previous orders)

The output dataset is structured to be consumed directly by a PyTorch model, with the following tensors:
- **impressions**: `[batch_size]`
- **actions**: `[batch_size, 1000]` (padded with 0s if fewer than 1000 actions)
- **action_types**: `[batch_size, 1000]` (1=click, 2=cart, 3=order, 0=missing)

## Pipeline
1. **Schema Standardization**  
   All input datasets are cast into a consistent schema with timestamps and standardized field names.

2. **Unification of Actions**  
   - Clicks → `action_type=1`  
   - Add-to-cart → `action_type=2`  
   - Previous orders → `action_type=3`

3. **Data Leakage Prevention**  
   Actions are filtered to include only those that occurred **strictly before** the impression date (`dt`).  
   This ensures the model does not peek into future actions.

4. **Sequence Building**  
   - For each `(customer_id, impression_date)` we collect up to **1000 most recent actions** in descending timestamp order.  
   - Padding is applied to ensure a fixed-length sequence.

5. **Join with Impressions**  
   Each impression row is exploded (one row per item in carousel) and joined with the corresponding action history.

6. **Final Output**  
   A Spark DataFrame with columns:
   - `impressions`
   - `actions`
   - `action_types`

## Performance Notes
- **Window functions** (`row_number`, `collect_list`) are used to efficiently rank and collect recent actions.
- **Broadcast join** is considered for impression ↔ actions join when the customer dimension fits in memory.
- At scale, further optimizations may include bucketing by `customer_id` or caching intermediate results.

## Running Locally
```bash
pip install -r requirements.txt
pytest tests/
python scripts/run_pipeline.py
