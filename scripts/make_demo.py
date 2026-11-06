#!/usr/bin/env python3
"""Проверка "на слух и на глаз": берёт пару треков из датасета, прогоняет
через обученную модель и кладёт рядом wav + midi + эталонный midi.

Нужен, чтобы после обучения в артефактах лежало что-то, что можно послушать.

    python scripts/make_demo.py --data data --model out/saved_model --out demo
"""

import argparse
import json
import os
import sys

import numpy as np
import pretty_midi
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SR  # noqa: E402


def write_reference_midi(notes, path):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=100, pitch=int(pitch),
            start=float(start), end=float(max(end, start + 0.05)),
        ))
    pm.instruments.append(inst)
    pm.write(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="out/saved_model")
    ap.add_argument("--out", default="demo")
    ap.add_argument("--count", type=int, default=2)
    args = ap.parse_args()

    from basic_pitch.inference import predict

    with open(os.path.join(args.data, "index.json")) as f:
        index = json.load(f)
    paths = (index.get("validation") or index["train"])[: args.count]

    os.makedirs(args.out, exist_ok=True)
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        with np.load(path) as z:
            audio = z["audio"].astype(np.float32) / 32767.0
            notes = z["notes"]

        wav_path = os.path.join(args.out, name + ".wav")
        sf.write(wav_path, audio, SR)
        write_reference_midi(notes, os.path.join(args.out, name + "_reference.mid"))

        try:
            _, midi_data, _ = predict(wav_path, model_or_model_path=args.model)
            out_mid = os.path.join(args.out, name + "_predicted.mid")
            midi_data.write(out_mid)
            print(f"[demo] {name}: {len(midi_data.instruments[0].notes)} нот -> {out_mid}")
        except Exception as e:
            print(f"[demo] модель не отработала на {name}: {e}")


if __name__ == "__main__":
    main()
