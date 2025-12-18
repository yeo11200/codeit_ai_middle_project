import os
import yaml
from typing import Dict, Any
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

class ConfigLoader:
    def __init__(self, config_path: str = None):
        if config_path is None:
            # 설정 경로가 주어지지 않으면 local.yaml을 기본값으로 사용하거나 환경 변수를 확인합니다.
            config_path = os.getenv("CONFIG_PATH", "config/local.yaml")
        
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """설정 파일(YAML)을 읽어옵니다."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        return config

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """
        점(.)으로 구분된 키를 사용하여 중첩된 설정 값을 가져옵니다.
        예: config.get("model.llm_name")
        """
        keys = key.split(".")
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

# 쉽게 접근할 수 있도록 싱글톤 인스턴스 생성
# 사용법: from src.common.config import config
try:
    config_loader = ConfigLoader()
    config = config_loader.config
except Exception as e:
    print(f"경고: 설정을 로드할 수 없습니다: {e}")
    config = {}
