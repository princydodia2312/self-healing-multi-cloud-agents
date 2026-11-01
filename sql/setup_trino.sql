CREATE SCHEMA IF NOT EXISTS hive.shop_orders;

DROP TABLE IF EXISTS hive.shop_orders.pipe_shop_orders_to_s3;

CREATE TABLE hive.shop_orders.pipe_shop_orders_to_s3 (
    order_id VARCHAR,
    order_date TIMESTAMP,
    customer_name VARCHAR,
    customer_email VARCHAR,
    customer_city VARCHAR,
    product VARCHAR,
    price DOUBLE,
    qty BIGINT,
    is_suspicious BOOLEAN,
    run_date DATE
) WITH (
    format = 'PARQUET',
    external_location = 's3://princy2312-shop-orders-insights-data/processed/',
    partitioned_by = ARRAY['run_date']
);
