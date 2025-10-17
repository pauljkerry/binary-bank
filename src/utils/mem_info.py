import json, subprocess


def free_ram_gib():
    try:
        cmd = [
            "powershell.exe", "-NoProfile", "-Command",
            "$o=Get-CimInstance Win32_OperatingSystem; "
            "$r=@{FreeGB=$o.FreePhysicalMemory/1KB/1024}; "
            "$r | ConvertTo-Json -Compress"
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        data = json.loads(r.stdout.strip())
        return data["FreeGB"]
    except Exception:
        return -1


def free_vram_gib():
    try:
        import pynvml as nvml
        nvml.nvmlInit()
        n = nvml.nvmlDeviceGetCount()
        frees = []
        for i in range(n):
            h = nvml.nvmlDeviceGetHandleByIndex(i)
            mem = nvml.nvmlDeviceGetMemoryInfo(h)  # bytes
            frees.append(mem.free / (1024 ** 3))   # -> GiB
        nvml.nvmlShutdown()
        return frees[0]
    except Exception:
        return -1