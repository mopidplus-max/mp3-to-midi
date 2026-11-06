#!/usr/bin/env python3
"""Сколько пар "аудио <-> ноты" уже сгенерировано.

    python scripts/count_pairs.py data
"""

import json
import os
import sys


def human(n):
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024 or unit == "ГБ":
            return f"{n:.1f} {unit}"
        n /= 1024


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    index_path = os.path.join(data_dir, "index.json")

    if not os.path.exists(index_path):
        print(f"нет {index_path} — генерация ещё не дошла до записи индекса")
        # но файлы на диске могут уже быть, посчитаем их
        npz = [f for f in os.listdir(data_dir) if f.endswith(".npz")] if os.path.isdir(data_dir) else []
        print(f"на диске .npz: {len(npz)}")
        return

    with open(index_path) as f:
        index = json.load(f)

    train = index.get("train", [])
    val = index.get("validation", [])
    size = sum(os.path.getsize(p) for p in train + val if os.path.exists(p))
    npz_on_disk = [f for f in os.listdir(data_dir) if f.endswith(".npz")] if os.path.isdir(data_dir) else []

    print(f"пар в индексе: {len(train) + len(val)}")
    print(f"  train: {len(train)}")
    print(f"  val:   {len(val)}")
    print(f"  на диске .npz: {len(npz_on_disk)}")
    print(f"  суммарный размер: {human(size)}")

    stats = os.path.join(data_dir, "gen_stats.json")
    if os.path.exists(stats):
        with open(stats) as f:
            g = json.load(f)
        print(f"  по gen_stats: {g.get('count')} треков за {g.get('minutes')} мин "
              f"(~{g.get('per_clip_sec')} с на трек)")


if __name__ == "__main__":
    main()
