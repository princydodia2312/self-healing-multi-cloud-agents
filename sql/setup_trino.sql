CREATE SCHEMA IF NOT EXISTS hive.shop_orders;

DROP TABLE IF EXISTS hive.shop_orders.pipe_shop_orders_to_s3;

CREATE TABLE hive.shop_orders.pipe_shop_orders_to_s3 (
    _id            VARCHAR,
    customer_city  VARCHAR,
    customer_email VARCHAR,
    customer_name  VARCHAR,
    order_date     TIMESTAMP,
    order_id       VARCHAR,
    price          DOUBLE,
    product        VARCHAR,
    qty            BIGINT,
    is_suspicious  BOOLEAN,
    run_date       DATE
) WITH (
    format            = 'PARQUET',
    external_location = 's3://princy2312-shop-orders-insights-data/processed/',
    partitioned_by    = ARRAY['run_date']
);
