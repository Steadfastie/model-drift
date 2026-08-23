import json
from dataclasses import dataclass, fields
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
    artifacts_dir: str = "artifacts"
    classification_threshold: float = 0.5
    clearml_project: str = "cross-model-drift"
    clearml_web_host: str = "http://localhost:8080"
    clearml_api_host: str = "http://localhost:8008"
    clearml_files_host: str = "http://localhost:8081"

    @property
    def mysql_uri(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    def artifacts_path(self) -> Path:
        path = Path(self.artifacts_dir)
        if not path.is_absolute():
            path = project_root() / path
        return path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(name: str = "local") -> AppConfig:
    """Load configs/{name}.json into an AppConfig."""
    config_path = project_root() / "configs" / f"{name}.json"
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    allowed = {item.name for item in fields(AppConfig)}
    return AppConfig(**{key: value for key, value in raw.items() if key in allowed})
