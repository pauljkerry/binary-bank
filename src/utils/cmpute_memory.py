# pip install psutil  # 入ってなければ
from contextlib import contextmanager
from time import perf_counter
import os


def _meminfo():
    # /proc/meminfo から MemTotal, MemAvailable を kB で読む
    total = avail = None
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) * 1024
            elif line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) * 1024
    return total, avail

def linux_vm_available_mb() -> int:
    total, avail = _meminfo()
    return int((avail or 0) / (1024*1024))

def linux_vm_total_mb() -> int:
    total, _ = _meminfo()
    return int((total or 0) / (1024*1024))

def cgroup_free_mb() -> int | None:
    # cgroup 上限が設定されていれば「上限-現在使用量」を返す（なければ None）
    try:
        # cgroup v2
        pmax = "/sys/fs/cgroup/memory.max"
        pcur = "/sys/fs/cgroup/memory.current"
        if os.path.exists(pmax) and os.path.exists(pcur):
            limit = open(pmax).read().strip()
            used  = int(open(pcur).read().strip())
            if limit != "max":
                limit = int(limit)
                return int((limit - used) / (1024*1024))
            return None
        # cgroup v1
        pmax = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        pcur = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
        if os.path.exists(pmax) and os.path.exists(pcur):
            limit = int(open(pmax).read().strip())
            used  = int(open(pcur).read().strip())
            return int((limit - used) / (1024*1024))
    except Exception:
        pass
    return None

def process_rss_mb() -> int | None:
    try:
        import psutil, os
        proc = psutil.Process(os.getpid())
        return int(proc.memory_info().rss / (1024*1024))
    except Exception:
        return None

def snapshot_mem():
    return {
        "vm_total_mb": linux_vm_total_mb(),
        "vm_avail_mb": linux_vm_available_mb(),   # ← “いまWSLで使える目安”
        "cgroup_free_mb": cgroup_free_mb(),       # ← 上限がある場合の残り
        "proc_rss_mb": process_rss_mb(),          # ← 現プロセスの実使用
    }

@contextmanager
def mem_watch(label: str = "", log: callable | None = None):
    before = snapshot_mem()
    t0 = perf_counter()
    yield
    after = snapshot_mem()
    dt = {k: (after[k] - before[k] if (before[k] is not None and after[k] is not None) else None)
          for k in after}
    msg = (f"[mem] {label} | avail {before['vm_avail_mb']}→{after['vm_avail_mb']} MB "
           f"({dt['vm_avail_mb']:+} MB), rss {before['proc_rss_mb']}→{after['proc_rss_mb']} MB "
           f"({(dt['proc_rss_mb'] if dt['proc_rss_mb'] is not None else 'NA')})")
    print(msg)
    if log:
        log({"mem/avail_mb": after["vm_avail_mb"],
             "mem/proc_rss_mb": after["proc_rss_mb"]})
