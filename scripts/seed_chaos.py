"""
Chaos seeder — loads deliberately dirty sales rows into the source database so the
agents' generated pipeline hits real data-quality problems (negative prices, NULL
quantities, future dates, unknown currencies) and the self-healing loop has
something to fix.

    python scripts/seed_chaos.py --target eu_sales --db-type postgres --rows 100
    python scripts/seed_chaos.py --target eu_sales --db-type sqlite   # local file in data/
"""
import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from faker import Faker
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import quote_plus

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.cloud_config import cloud_get  # noqa: E402

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load Environment Variables
env_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Loaded environment variables from: {env_path}")

fake = Faker()

TARGETS = ["eu_sales"]


def create_chaos_data(target, n_rows=100):
    """Generate a DataFrame with intentional anomalies for the given target."""
    logger.info(f"Generating chaos data for target: {target}")

    if target == "eu_sales":
        data_list = []
        for _ in range(n_rows):
            data_list.append({
                "order_id": fake.uuid4(),
                "unit_price": np.random.choice([fake.pyfloat(left_digits=2, right_digits=2, positive=True), -10.0, 0.0, None], p=[0.8, 0.05, 0.05, 0.1]),
                "quantity": np.random.choice([np.random.randint(1, 10), -1, None], p=[0.8, 0.1, 0.1]),
                "order_date": np.random.choice([fake.date_this_year(), "2030-01-01", None], p=[0.8, 0.1, 0.1]),
                "currency": np.random.choice(["EUR", "GBP", "XYZ", None], p=[0.7, 0.1, 0.1, 0.1]),
            })
        return pd.DataFrame(data_list)

    return pd.DataFrame()


def build_engine(db_type, target="eu_sales"):
    """
    Create a SQLAlchemy engine for the source database.
      sqlite   → data/<target>_raw.db (local, no cloud)
      postgres → AWS RDS PostgreSQL, credentials via cloud_get (SSM → bootstrap file → POSTGRES_DB_*)
      mysql    → MySQL source, credentials via cloud_get(db_type="mysql") (MYSQL_DB_*)
    """
    if db_type == "sqlite":
        db_path = os.path.join("data", f"{target}_raw.db")
        return create_engine(f"sqlite:///{db_path}")

    if db_type == "postgres":
        driver, default_port = "postgresql+psycopg2", "5432"
    elif db_type == "mysql":
        driver, default_port = "mysql+pymysql", "3306"
    else:
        raise ValueError(f"Unsupported DB type: {db_type}")

    host = cloud_get("aws", "db_host", db_type=db_type)
    port = cloud_get("aws", "db_port", db_type=db_type) or default_port
    user = cloud_get("aws", "db_user", db_type=db_type)
    password = cloud_get("aws", "db_password", db_type=db_type)
    db_name = cloud_get("aws", "db_name", db_type=db_type)

    # URL-encode user/password: a special char (@ : / # %) in the password otherwise
    # breaks SQLAlchemy URL parsing and the wrong password is sent → "Access denied".
    url = f"{driver}://{quote_plus(user or '')}:{quote_plus(password or '')}@{host}:{port}/{db_name}"
    return create_engine(url)


def _replace_table(engine, df: pd.DataFrame, table_name: str, what: str) -> None:
    """Write df as table_name (REPLACE), logging how many rows are being dropped first."""
    with engine.connect() as _c:  # fail fast on a bad connection before to_sql
        try:
            _prev = _c.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()
            logger.warning(
                f"♻️  REPLACING '{table_name}': dropping {_prev} existing row(s) with {len(df)} "
                f"{what} rows. Destructive by design (source reset)."
            )
        except Exception:
            logger.info(f"'{table_name}' has no prior rows / does not exist yet — creating it.")
    df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)


def seed_dataframe_to_source(df: pd.DataFrame, slug: str, cloud: str = "aws") -> str:
    """
    Load a user-supplied sample ("bring your own data" via the Streamlit UI) into the AWS
    source database (the eu_sales RDS) as `raw_<slug>`, so a pipeline whose source_table is
    `raw_<slug>` has a real table to extract at deploy time. Returns the table name written.
    """
    if df is None or df.empty:
        raise ValueError("Empty dataset — nothing to seed into the source database.")
    if cloud != "aws":
        raise ValueError(f"Unsupported cloud for seeding: {cloud!r} (only 'aws').")
    table_name = f"raw_{slug}"
    _replace_table(build_engine("postgres"), df, table_name, "uploaded sample")
    logger.info(f"✅ Seeded {len(df)} rows into source table '{table_name}' (aws)")
    return table_name


def seed_chaos(target_arg, db_type, n_rows):
    targets_to_run = TARGETS if target_arg == "all" else [target_arg]

    for target in targets_to_run:
        try:
            df = create_chaos_data(target, n_rows=n_rows)
            if df.empty:
                logger.warning(f"⚠️ No data generated for target {target}")
                continue
            table_name = f"raw_{target}"
            logger.info(f"🔥 Injecting {n_rows} chaotic rows into '{table_name}' ({db_type})...")
            _replace_table(build_engine(db_type, target), df, table_name, "chaos")
            logger.info(f"✅ Successfully seeded '{table_name}'")
        except Exception as e:
            logger.error(f"❌ FATAL ERROR for {target}: {str(e)}")
            # Stop execution so the error is visible in GitHub Actions
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chaos data seeder for the self-healing demo")
    parser.add_argument("--target", choices=TARGETS + ["all"], default="eu_sales",
                        help="Pipeline whose source table to seed")
    parser.add_argument("--db-type", choices=["sqlite", "postgres", "mysql"], default="postgres",
                        help="Database engine to use")
    parser.add_argument("--rows", type=int, default=100, help="Number of rows to generate")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    logger.info(f"🚀 STARTING CHAOS: Scope={args.target}, DB={args.db_type}, Rows={args.rows}")
    seed_chaos(args.target, args.db_type, args.rows)
    logger.info("✨ Process Completed.")
