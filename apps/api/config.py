import os
from pathlib import Path
def load_local_config():
    path=Path(__file__).resolve().parents[2]/'.env'
    if not path.exists():return
    for line in path.read_text(encoding='utf8').splitlines():
        if '=' not in line or line.strip().startswith('#'):continue
        key,value=line.split('=',1);key=key.strip();value=value.strip().strip('"').strip("'")
        if key.startswith(('DEEPSEEK_','GESTUREFLOW_','AI_','ASR_')):os.environ.setdefault(key,value)
