import os
import yaml
from dotenv import load_dotenv

load_dotenv()

class ConfigLoader:
    def __init__(self, config_path: str = "config/local.yaml"):
        self.config = self._load_config(config_path)

    def _load_config(self, path):
        if not os.path.exists(path):
            # 파일이 없으면 빈 딕셔너리 반환 혹은 에러 처리
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, key, default=None):
        keys = key.split(".")
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except:
            return default

# 전역 객체 생성
try:
    config = ConfigLoader().config
except:
    config = {}

