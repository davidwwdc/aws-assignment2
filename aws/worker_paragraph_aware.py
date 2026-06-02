from __future__ import annotations

from config import PARAGRAPH_AWARE_QUEUE_URL, require_worker_config
from worker_common import run_worker


if __name__ == "__main__":
    require_worker_config()
    run_worker("paragraph_aware", PARAGRAPH_AWARE_QUEUE_URL)
