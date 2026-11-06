#!/usr/bin/env python3
"""Готовит датасет для обучения: MIDI -> аудио (fluidsynth) -> .npz

Почему синтез, а не реальные записи: настоящий MAESTRO с аудио весит ~108 ГБ,
на GitHub-раннере столько места нет. Мы берём MIDI-часть MAESTRO (57 МБ,
реальные живые исполнения с настоящей динамикой и темпом), рендерим её
в аудио саундфонтом и получаем бесконечные идеально выровненные пары
"аудио <-> ноты". Модель выучит пианино, но не научится понимать,
скажем, гитару или электронику — про это честно написано в README.

Пример:
    python scripts/prepare_data.py --out data --tracks 60 --max-seconds 120
"""

import argparse
import csv
import io
import json
import os
import random
import shutil
import subprocess
import sys
import urllib.request
import zipfile

import numpy as np
import pretty_midi
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SR  # noqa: E402

MAESTRO_MIDI_ZIP = (
    "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/"
    "maestro-v3.0.0-midi.zip"
)

SOUNDFONT_CANDIDATES = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/FluidR3_GS.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
]


def find_soundfont(explicit=None):
    if explicit:
        return explicit
    for p in SOUNDFONT_CANDIDATES:
        if os.path.exists(p):
            return p
    raise SystemExit(
        "Не найден саундфонт. Поставь его: sudo apt-get install -y fluid-soundfont-gm\n"
        "или укажи путь вручную через --soundfont"
    )


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"[data] уже скачано: {dest}")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[data] качаю {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"[data] готово: {dest} ({os.path.getsize(dest)/1e6:.1f} МБ)")
    return dest


def unpack_maestro(zip_path, workdir):
    marker = os.path.join(workdir, ".unpacked")
    if os.path.exists(marker):
        return workdir
    print("[data] распаковываю MIDI...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(workdir)
    # внутри обычно папка maestro-v3.0.0/
    entries = [e for e in os.listdir(workdir) if os.path.isdir(os.path.join(workdir, e))]
    root = workdir
    if entries and any(e.startswith("maestro") for e in entries):
        root = os.path.join(workdir, [e for e in entries if e.startswith("maestro")][0])
    open(marker, "w").close()
    return root


def read_maestro_csv(root):
    """Возвращает список (midi_path, split)."""
    csv_path = None
    for name in os.listdir(root):
        if name.endswith(".csv"):
            csv_path = os.path.join(root, name)
            break
    if csv_path is None:
        raise SystemExit(f"Не нашёл CSV с метаданными в {root}")

    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            items.append((os.path.join(root, row["midi_filename"]), row["split"]))
    return items


def render(midi_path, wav_path, soundfont, timeout=180):
    cmd = [
        "fluidsynth", "-ni", "-F", wav_path,
        "-r", str(SR), "-g", "0.9",
        soundfont, midi_path,
    ]
    subprocess.run(cmd, check=True, timeout=timeout,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def midi_notes(path):
    pm = pretty_midi.PrettyMIDI(str(path))
    out = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            out.append((n.pitch, n.start, n.end))
    out.sort(key=lambda x: x[1])
    return np.asarray(out, dtype=np.float32).reshape(-1, 3)


def save_track(npz_path, wav_path, notes, max_seconds, rng):
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        raise RuntimeError(f"{wav_path}: {sr} Гц вместо {SR}, fluidsynth не послушался -r")

    duration = len(audio) / SR
    if max_seconds and duration > max_seconds:
        # берём случайный кусок — так модель видит разные места произведения
        start = rng.uniform(0, duration - max_seconds)
        i = int(start * SR)
        keep = int(max_seconds * SR)
        audio = audio[i:i + keep]
        shift = i / SR
        keep_notes = notes[(notes[:, 1] >= shift) & (notes[:, 1] < shift + max_seconds)]
        notes = keep_notes.copy()
        notes[:, 1] -= shift
        notes[:, 2] -= shift

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio = audio / max(peak, 0.5)  # не даём клиппинга, но и не задираем тишину

    np.savez(
        npz_path,
        audio=(audio * 32767.0).astype(np.int16),
        notes=notes.astype(np.float32),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="куда сложить датасет")
    ap.add_argument("--work", default=".cache/data", help="кэш загрузок")
    ap.add_argument("--tracks", type=int, default=60, help="сколько треков train (validation = 15%)")
    ap.add_argument("--max-seconds", type=float, default=120.0, help="сколько секунд брать с трека")
    ap.add_argument("--soundfont", default=None)
    ap.add_argument("--midi-dir", default=None,
                    help="свои MIDI вместо MAESTRO (папка с .mid/.midi)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    if args.midi_dir:
        items = []
        for root, _, files in os.walk(args.midi_dir):
            for f in sorted(files):
                if f.lower().endswith((".mid", ".midi")):
                    items.append((os.path.join(root, f), "train"))
    else:
        zip_path = download(MAESTRO_MIDI_ZIP, os.path.join(args.work, "maestro-v3.0.0-midi.zip"))
        root = unpack_maestro(zip_path, args.work)
        items = read_maestro_csv(root)

    train_items = [i for i in items if i[1] == "train"]
    val_items = [i for i in items if i[1] != "train"]
    rng.shuffle(train_items)
    rng.shuffle(val_items)

    n_val = max(4, int(args.tracks * 0.15))
    chosen = train_items[: args.tracks]
    val_chosen = val_items[:n_val] if val_items else train_items[args.tracks: args.tracks + n_val]

    soundfont = find_soundfont(args.soundfont)
    print(f"[data] саундфонт: {soundfont}")
    print(f"[data] треков: train={len(chosen)}, val={len(val_chosen)}")

    index = {"train": [], "validation": []}
    tmp_wav = os.path.join(args.work, "_tmp.wav")

    for split, group in (("train", chosen), ("validation", val_chosen)):
        for k, (midi_path, _) in enumerate(group):
            name = f"{split}_{k:04d}"
            npz_path = os.path.join(args.out, name + ".npz")
            if os.path.exists(npz_path):
                index[split].append(npz_path)
                continue
            try:
                notes = midi_notes(midi_path)
                if len(notes) < 10:
                    continue
                render(midi_path, tmp_wav, soundfont)
                save_track(npz_path, tmp_wav, notes, args.max_seconds, rng)
                index[split].append(npz_path)
                print(f"[data] {name}: {os.path.basename(midi_path)} "
                      f"({len(notes)} нот, {os.path.getsize(npz_path)/1e6:.1f} МБ)")
            except Exception as e:  # один битый трек не должен убивать всю подготовку
                print(f"[data] пропуск {midi_path}: {e}")

    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    total = sum(os.path.getsize(p) for s in index.values() for p in s) / 1e9
    print(f"[data] готово: {len(index['train'])} train / {len(index['validation'])} val, {total:.2f} ГБ")


if __name__ == "__main__":
    main()
