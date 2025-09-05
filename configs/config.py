# config.py

import yaml

class Config:
    def __init__(self, path: str):
        with open(path, 'r') as f:
            self._c = yaml.safe_load(f)

    def __getattr__(self, name):
        try:
            return self._c[name]
        except KeyError:
            raise AttributeError(f"No such config key: {name}") from None
