from datetime import timedelta

def format_ass_time(seconds: float) -> str:
    # ASS time format: H:MM:SS.cs (cs is centiseconds)
    td = timedelta(seconds=seconds)
    total_sec = int(td.total_seconds())
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    centisecs = int((seconds - int(seconds)) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

print(format_ass_time(1.234))
print(format_ass_time(3601.05))
