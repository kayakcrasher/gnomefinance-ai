from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return here.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_find_env()), extra="ignore")

    google_api_key: str = ""
    database_path: str = "./gnomefinance.db"
    log_level: str = "INFO"


settings = Settings()
