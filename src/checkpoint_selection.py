"""Prefer newly trained native runs over the initially imported MJLab policy."""
from pathlib import Path
import re

def latest_checkpoint(logs):
    candidates=[p for p in Path(logs).glob('*/model_*.pt')
                if 'smoke' not in str(p.parent) and re.fullmatch(r'model_\d+\.pt',p.name)]
    native=[p for p in candidates if (p.parent/'training_config.json').is_file()]
    pool=native or candidates
    if not pool:raise FileNotFoundError('No saved Playground checkpoints or imported policy')
    run=max({p.parent for p in pool},key=lambda p:p.stat().st_mtime_ns)
    model=max((p for p in pool if p.parent==run),key=lambda p:int(p.stem.split('_')[-1]))
    return model,'latest native training run' if native else 'imported MJLab policy fallback'
