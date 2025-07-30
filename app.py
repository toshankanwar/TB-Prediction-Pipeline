import os
import datetime
import pandas as pd
import numpy as np
import requests
from firebaseAdmin import initialize_firebase
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv

load_dotenv()

db = initialize_firebase()
PREDICTION_API_URL = os.getenv('PREDICTION_API_URL')


def fetch_latest_orders(since_datetime):
    """
    Fetch all documents from Firestore 'orders' collection,
    flatten nested 'items' array so each row corresponds to an order item with 
    order-level metadata merged (including Order ID as Firestore doc id),
    filter locally by parsed 'createdAt' datetime string.
    """
    docs = db.collection('orders').stream()
    records = []

    for doc in docs:
        rec = doc.to_dict()
        order_meta = {k: v for k, v in rec.items() if k != 'items'}
        order_meta['Order ID'] = doc.id  # Add Firestore doc ID as Order ID
        items = rec.get('items', [])

        # Flatten each item: one record per item plus order metadata
        for item in items:
            flat_record = {**order_meta, **item}
            records.append(flat_record)

    if not records:
        print("No order data found in Firestore.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Parse 'createdAt' string to datetime (UTC-aware)
    df['createdAt_dt'] = pd.to_datetime(df['createdAt'], utc=True, errors='coerce')

    # Filter locally based on since_datetime (timezone-aware)
    filtered_df = df[df['createdAt_dt'] > since_datetime].copy()
    print(f"Fetched {len(df)} order items total; {len(filtered_df)} after filtering by date > {since_datetime}")

    if filtered_df.empty:
        print(f"No new order data after {since_datetime}")
    return filtered_df


def filter_out_pending_cancelled(df):
    """
    Filter out rows where:
    - 'Order Status' (a list) contains both 'pending' AND 'cancelled' (exclude such rows),
    - OR 'Payment Status' (a string) contains 'cancelled' (exclude such rows),
    - but allow 'pending' in 'Payment Status'.
    """
    def has_both_pending_cancelled(status_list):
        if not isinstance(status_list, list):
            return False  # Keep row if not a list
        lowered = [s.lower() for s in status_list if isinstance(s, str)]
        return ('pending' in lowered) and ('cancelled' in lowered)

    mask_order_status_ok = ~df['Order Status'].apply(has_both_pending_cancelled)

    def payment_status_not_cancelled(status_str):
        if not isinstance(status_str, str):
            return False  # Exclude non-string payment statuses for safety
        status = status_str.lower()
        return 'cancelled' not in status

    mask_payment_status_ok = df['Payment Status'].apply(payment_status_not_cancelled)

    return df[mask_order_status_ok & mask_payment_status_ok].copy()


def clean_data(df):
    if df.empty:
        return df

    # Use parsed datetime column as 'Order Date'
    df['Order Date'] = df['createdAt_dt']

    # Rename Firestore keys to consistent column names
    rename_map = {}

    if 'orderStatus' in df.columns:
        rename_map['orderStatus'] = 'Order Status'
    elif 'order_status' in df.columns:
        rename_map['order_status'] = 'Order Status'

    if 'paymentStatus' in df.columns:
        rename_map['paymentStatus'] = 'Payment Status'
    elif 'payment_status' in df.columns:
        rename_map['payment_status'] = 'Payment Status'

    if 'name' in df.columns:
        rename_map['name'] = 'Item Name'
    if 'quantity' in df.columns:
        rename_map['quantity'] = 'Quantity'
    if 'price' in df.columns:
        rename_map['price'] = 'Price'
    if 'total' in df.columns:
        rename_map['total'] = 'Total Price'
    if 'subtotal' in df.columns:
        rename_map['subtotal'] = 'Subtotal'

    df = df.rename(columns=rename_map)

    # Filter out rows with pending/cancelled status if columns exist
    if 'Order Status' in df.columns and 'Payment Status' in df.columns:
        df = filter_out_pending_cancelled(df)
    else:
        print("Warning: 'Order Status' or 'Payment Status' columns missing. Skipping status filtering.")

    # Ensure Quantity column exists and is numeric
    if 'Quantity' not in df.columns:
        raise KeyError("'Quantity' column missing after renaming!")

    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)

    # Calculate Total Price if missing
    if 'Total Price' in df.columns:
        df['Total Price'] = pd.to_numeric(df['Total Price'], errors='coerce').fillna(0)
    elif 'Price' in df.columns:
        df['Total Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0) * df['Quantity']
    else:
        df['Total Price'] = 0

    # Drop rows with missing crucial fields
    df = df.dropna(subset=['Order Date', 'Item Name'])
    return df


def feature_engineering_product_level(daily_prod):
    daily_prod['day_of_week'] = daily_prod['Order Date'].dt.dayofweek
    daily_prod['day'] = daily_prod['Order Date'].dt.day
    daily_prod['month'] = daily_prod['Order Date'].dt.month
    daily_prod['year'] = daily_prod['Order Date'].dt.year
    daily_prod['is_weekend'] = daily_prod['day_of_week'].isin([5, 6]).astype(int)
    daily_prod['lag_1_qty'] = daily_prod.groupby('Item Name')['daily_quantity_sold'].shift(1).fillna(0)
    daily_prod['lag_7_qty'] = daily_prod.groupby('Item Name')['daily_quantity_sold'].shift(7).fillna(0)
    daily_prod['lag_1_orders'] = daily_prod.groupby('Item Name')['daily_order_count'].shift(1).fillna(0)
    daily_prod['lag_7_orders'] = daily_prod.groupby('Item Name')['daily_order_count'].shift(7).fillna(0)
    daily_prod['roll_3d_qty'] = daily_prod.groupby('Item Name')['daily_quantity_sold'].transform(lambda x: x.shift(1).rolling(3).mean()).fillna(0)
    daily_prod['roll_7d_qty'] = daily_prod.groupby('Item Name')['daily_quantity_sold'].transform(lambda x: x.shift(1).rolling(7).mean()).fillna(0)
    return daily_prod


def prepare_data_for_modeling(df_clean):
    daily_prod = df_clean.groupby(['Order Date', 'Item Name']).agg(
        daily_quantity_sold=('Quantity', 'sum'),
        daily_order_count=('Order ID', 'nunique'),
        daily_revenue=('Total Price', 'sum')
    ).reset_index().sort_values(['Item Name', 'Order Date'])

    daily_prod_fe = feature_engineering_product_level(daily_prod)

    daily_prod_fe['target_next_day_quantity'] = daily_prod_fe.groupby('Item Name')['daily_quantity_sold'].shift(-1)
    daily_prod_fe['target_next_day_orders'] = daily_prod_fe.groupby('Item Name')['daily_order_count'].shift(-1)

    daily_prod_fe = daily_prod_fe.dropna(subset=['target_next_day_quantity', 'target_next_day_orders'])

    daily_total = daily_prod_fe.groupby('Order Date').agg(
        total_quantity=('daily_quantity_sold', 'sum'),
        total_orders=('daily_order_count', 'sum'),
        target_next_day_total_quantity=('target_next_day_quantity', 'sum'),
        target_next_day_total_orders=('target_next_day_orders', 'sum')
    ).reset_index()

    daily_total['lag_1_total_qty'] = daily_total['total_quantity'].shift(1).fillna(0)
    daily_total['lag_7_total_qty'] = daily_total['total_quantity'].shift(7).fillna(0)
    daily_total['lag_1_total_orders'] = daily_total['total_orders'].shift(1).fillna(0)
    daily_total['lag_7_total_orders'] = daily_total['total_orders'].shift(7).fillna(0)

    daily_total['day_of_week'] = daily_total['Order Date'].dt.dayofweek
    daily_total['is_weekend'] = daily_total['day_of_week'].isin([5, 6]).astype(int)

    return daily_prod_fe, daily_total


def train_model(X, y, label="Model"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{label} MAE: {mae:.2f}")
    return model


def post_predictions_to_api(predictions_df, prediction_type):
    if predictions_df.empty:
        print(f"No predictions to post for {prediction_type}")
        return

    for _, row in predictions_df.iterrows():
        payload = {
            "prediction_type": prediction_type,
            "date": str(row['Order Date'].date()),
            "item_name": row.get('Item Name', None),
            "predicted_value": row['prediction']
        }
        try:
            response = requests.post(PREDICTION_API_URL, json=payload, timeout=10)
            if response.status_code != 200:
                print(f"Error posting prediction: {response.text}")
        except Exception as e:
            print(f"Exception during posting prediction: {e}")


def main():
    print(f"Running pipeline at {datetime.datetime.utcnow().isoformat()} UTC")

    fetch_since = (datetime.datetime.utcnow() - datetime.timedelta(days=14)).replace(tzinfo=datetime.timezone.utc)

    raw_df = fetch_latest_orders(fetch_since)
    if raw_df.empty:
        print("No new data fetched, exiting pipeline.")
        return

    clean_df = clean_data(raw_df)

    item_df, agg_df = prepare_data_for_modeling(clean_df)

    item_features = [
        'daily_quantity_sold', 'daily_order_count', 'daily_revenue',
        'day_of_week', 'day', 'month', 'year', 'is_weekend',
        'lag_1_qty', 'lag_7_qty', 'lag_1_orders', 'lag_7_orders',
        'roll_3d_qty', 'roll_7d_qty'
    ]
    X_item = item_df[item_features]
    y_item_qty = item_df['target_next_day_quantity']
    model_item = train_model(X_item, y_item_qty, "Item Next Day Quantity Prediction")

    agg_features = [
        'total_orders', 'total_quantity',
        'lag_1_total_orders', 'lag_7_total_orders',
        'lag_1_total_qty', 'lag_7_total_qty',
        'day_of_week', 'is_weekend'
    ]
    X_agg = agg_df[agg_features]

    model_agg_orders = train_model(X_agg, agg_df['target_next_day_total_orders'], "Aggregate Next Day Orders Prediction")
    model_agg_qty = train_model(X_agg, agg_df['target_next_day_total_quantity'], "Aggregate Next Day Quantity Prediction")

    # Prediction date is always tomorrow based on current UTC date
    today_utc = datetime.datetime.utcnow().date()
    next_day = pd.Timestamp(today_utc + datetime.timedelta(days=1))
    print(f"Generating predictions for next day: {next_day.date()}")

    predictions = []
    for item_name in item_df['Item Name'].unique():
        last_row = item_df[item_df['Item Name'] == item_name].sort_values('Order Date').iloc[-1]
        input_data = pd.DataFrame([last_row[item_features].values], columns=item_features)
        pred_val = model_item.predict(input_data)[0]
        predictions.append({'Order Date': next_day, 'Item Name': item_name, 'prediction': pred_val})
    pred_df_item = pd.DataFrame(predictions)

    post_predictions_to_api(pred_df_item, 'item_quantity')

    last_agg_row = agg_df.sort_values('Order Date').iloc[-1]
    input_agg = pd.DataFrame([last_agg_row[agg_features].values], columns=agg_features)
    pred_orders = model_agg_orders.predict(input_agg)[0]
    pred_quantity = model_agg_qty.predict(input_agg)[0]

    post_predictions_to_api(pd.DataFrame([{'Order Date': next_day, 'prediction': pred_orders}]), 'total_orders')
    post_predictions_to_api(pd.DataFrame([{'Order Date': next_day, 'prediction': pred_quantity}]), 'total_quantity')

    print("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
