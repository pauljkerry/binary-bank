import subprocess, json


def win_avail_gb():
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
        return None