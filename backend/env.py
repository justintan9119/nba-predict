from pathlib import Path
import os


def load_local_env():
    base_path = Path(__file__).with_name('.env')
    local_path = Path(__file__).with_name('.env.local')
    initial_keys = set(os.environ.keys())

    def load_file(path):
        if not path.exists():
            return
        for raw_line in path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in initial_keys:
                os.environ[key] = value

    load_file(base_path)
    load_file(local_path)


load_local_env()
