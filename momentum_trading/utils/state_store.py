"""
State Store

Durable storage for bars, signals, weights, orders, fills,
funding ledger, equity curve, and parameter versions.
"""

import json
import pickle
from pathlib import Path
from datetime import datetime


class StateStore:
    """
    Persistent state management.
    
    Stores:
    - Historical bars (cache)
    - Signal history
    - Weight history
    - Order/fill log
    - Funding ledger (hourly)
    - Equity curve
    - Configuration snapshots
    """
    
    def __init__(self, data_dir: str = "data/state"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def save_snapshot(self, name: str, data: dict):
        """Save a state snapshot."""
        path = self.data_dir / f"{name}_{datetime.utcnow().isoformat()}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load_latest_snapshot(self, name: str) -> dict:
        """Load most recent snapshot."""
        pattern = f"{name}_*.json"
        files = sorted(self.data_dir.glob(pattern), reverse=True)
        
        if not files:
            return {}
        
        with open(files[0], "r") as f:
            return json.load(f)
    
    def append_jsonl(self, name: str, record: dict):
        """Append a record to a JSONL log file (orders, fills, equity)."""
        path = self.data_dir / f"{name}.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def read_jsonl(self, name: str, limit: int = 1000) -> list:
        """Read up to 'limit' records from end of a JSONL log file."""
        path = self.data_dir / f"{name}.jsonl"
        if not path.exists():
            return []
        with open(path, "r") as f:
            lines = f.readlines()
        return [json.loads(x) for x in lines[-limit:]]
