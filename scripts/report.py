#!/usr/bin/env python3
"""Кладёт итоги обучения в GitHub Step Summary (и просто печатает в лог)."""

import json
import os
import sys


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out/metrics.json"
    if not os.path.exists(path):
        print("[report] нет metrics.json — обучение не дошло до конца")
        return

    with open(path) as f:
        m = json.load(f)

    lines = [
        "### Обучение закончено",
        "",
        f"- минут обучения: **{m.get('minutes_used')}**",
        f"- эпох: **{m.get('steps')}**",
        f"- loss: **{m.get('final_loss')}** / val_loss: **{m.get('final_val_loss')}**",
        f"- batch: {m.get('batch_size')}, lr: {m.get('learning_rate')}",
        "",
        "Модель: `out/saved_model`. Демо (wav + midi) — в `demo/`.",
    ]
    text = "\n".join(lines)
    print(text)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
