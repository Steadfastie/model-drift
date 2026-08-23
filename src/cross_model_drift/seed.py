from pathlib import Path

import pymysql

from cross_model_drift.config import AppConfig, project_root

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS `{table}` (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    sender_id CHAR(12) NOT NULL,
    created DATETIME(3) NOT NULL,
    payout_country CHAR(2) NOT NULL,
    payout_currency CHAR(3) NOT NULL,
    amount_usd DECIMAL(12, 2) NOT NULL,
    fee_usd DECIMAL(12, 2) NOT NULL,
    status VARCHAR(32) NOT NULL,
    anti_fraud_status VARCHAR(16) NOT NULL,
    compliance_status VARCHAR(16) NOT NULL,
    PRIMARY KEY (id),
    KEY idx_transactions_created (created),
    KEY idx_transactions_sender_created (sender_id, created)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""


def connect(config: AppConfig) -> pymysql.connections.Connection:
    return pymysql.connect(
        host=config.mysql_host,
        port=config.mysql_port,
        user=config.mysql_user,
        password=config.mysql_password,
        database=config.mysql_database,
        charset="utf8mb4",
        local_infile=True,
        autocommit=True,
    )


def resolve_csv_path(config: AppConfig, csv_path: str | Path | None = None) -> Path:
    path = Path(csv_path) if csv_path is not None else Path(config.transactions_csv)
    if not path.is_absolute():
        path = project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"transactions CSV not found: {path}")
    return path.resolve()


def create_transactions_table(config: AppConfig) -> None:
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL.format(table=config.transactions_table))


def count_transactions(config: AppConfig) -> int:
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM `{config.transactions_table}`")
        row = cur.fetchone()
    return int(row[0]) if row else 0


def seed_transactions(
    config: AppConfig,
    csv_path: str | Path | None = None,
    *,
    replace: bool = False,
) -> int:
    """Load transactions.csv into MySQL via LOAD DATA LOCAL INFILE."""
    path = resolve_csv_path(config, csv_path)
    table = config.transactions_table
    create_transactions_table(config)

    with connect(config) as conn, conn.cursor() as cur:
        if replace:
            cur.execute(f"TRUNCATE TABLE `{table}`")

        existing = count_transactions(config)
        if existing and not replace:
            return existing

        escaped = path.as_posix().replace("\\", "\\\\").replace("'", "\\'")
        cur.execute(
            f"""
            LOAD DATA LOCAL INFILE '{escaped}'
            INTO TABLE `{table}`
            FIELDS TERMINATED BY ','
            OPTIONALLY ENCLOSED BY '"'
            LINES TERMINATED BY '\\n'
            IGNORE 1 LINES
            (
                sender_id,
                created,
                payout_country,
                payout_currency,
                amount_usd,
                fee_usd,
                status,
                anti_fraud_status,
                compliance_status
            )
            """
        )
    return count_transactions(config)
