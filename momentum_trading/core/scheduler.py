"""
Scheduler: Triggers rebalances at fixed intervals.

Manages timing for daily (00:05 UTC) or 4h (+5min after bar close) rebalances.
Handles funding-hour boundaries and prevents rebalance overlap.
"""

import time
from datetime import datetime, timezone
from typing import Callable

from momentum_trading.core.config import Config


class Scheduler:
    """
    Rebalance scheduler with precise timing control.
    
    Triggers rebalance callback at configured intervals:
    - Daily: at 00:05:00 UTC
    - 4h: at 00:05, 04:05, 08:05, 12:05, 16:05, 20:05 UTC
    
    Avoids rebalancing exactly on funding timestamps (hourly on the hour)
    to reduce noisy flips from premium spikes.
    """
    
    def __init__(self, config: Config, rebalance_callback: Callable[[], None]):
        """
        Initialize scheduler.
        
        Args:
            config: System configuration
            rebalance_callback: Function to call on each rebalance trigger
        """
        self.config = config
        self.rebalance_callback = rebalance_callback
        self.running = False
        self.last_rebalance_ts: float = 0.0
    
    def next_rebalance_time(self) -> datetime:
        """
        Calculate next rebalance timestamp.
        
        Returns:
            Next rebalance datetime (UTC)
        """
        now = datetime.now(timezone.utc)
        
        if self.config.bars.interval == "1d":
            # Daily at 00:05:00 UTC
            target = now.replace(hour=0, minute=5, second=0, microsecond=0)
            if now >= target:
                # Already past today's rebalance, schedule tomorrow
                from datetime import timedelta
                target += timedelta(days=1)
            return target
        
        elif self.config.bars.interval == "4h":
            # Every 4h at :05 past the hour
            # Valid hours: 00, 04, 08, 12, 16, 20
            valid_hours = [0, 4, 8, 12, 16, 20]
            current_hour = now.hour
            
            # Find next valid hour
            next_hour = None
            for h in valid_hours:
                if h > current_hour or (h == current_hour and now.minute < 5):
                    next_hour = h
                    break
            
            if next_hour is None:
                # Wrap to tomorrow 00:05
                from datetime import timedelta
                target = now.replace(hour=0, minute=5, second=0, microsecond=0)
                target += timedelta(days=1)
            else:
                target = now.replace(hour=next_hour, minute=5, second=0, microsecond=0)
            
            return target
        
        else:
            raise ValueError(f"Unsupported interval: {self.config.bars.interval}")
    
    def seconds_until_next_rebalance(self) -> float:
        """Calculate seconds until next rebalance."""
        next_time = self.next_rebalance_time()
        now = datetime.now(timezone.utc)
        delta = (next_time - now).total_seconds()
        return max(0.0, delta)
    
    def run_forever(self):
        """
        Run scheduler loop indefinitely.
        
        Blocks until stopped. Calls rebalance_callback at each trigger.
        """
        self.running = True
        print(f"[Scheduler] Started. Interval={self.config.bars.interval}")
        
        while self.running:
            try:
                # Calculate sleep time
                sleep_sec = self.seconds_until_next_rebalance()
                
                if sleep_sec > 0:
                    next_time = self.next_rebalance_time()
                    print(f"[Scheduler] Next rebalance in {sleep_sec:.0f}s at {next_time.isoformat()}")
                    time.sleep(min(sleep_sec, 60))  # Wake up every minute to check
                    continue
                
                # Time to rebalance
                now = datetime.now(timezone.utc)
                print(f"[Scheduler] Triggering rebalance at {now.isoformat()}")
                
                try:
                    self.rebalance_callback()
                    self.last_rebalance_ts = time.time()
                except Exception as e:
                    print(f"[Scheduler] ERROR during rebalance: {e}")
                    # Continue running despite errors
                
                # Sleep briefly to avoid double-trigger
                time.sleep(10)
            
            except KeyboardInterrupt:
                print("[Scheduler] Interrupted by user")
                self.running = False
                break
            
            except Exception as e:
                print(f"[Scheduler] ERROR in main loop: {e}")
                time.sleep(60)
    
    def stop(self):
        """Stop the scheduler loop."""
        print("[Scheduler] Stopping...")
        self.running = False

