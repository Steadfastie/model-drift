import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str
    mysql_database: str
    transactions_table: str
    transactions_csv: str

    @property
    def mysql_uri(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(name: str = "local") -> AppConfig:
    """Load configs/{name}.json into an AppConfig."""
    config_path = project_root() / "configs" / f"{name}.json"
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    return AppConfig(**raw)
