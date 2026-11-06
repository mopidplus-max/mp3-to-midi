#!/usr/bin/env python3
"""Генератор обучающих пар "аудио <-> ноты" прямо на раннере.

Всё случайное на каждом запуске:
  * стиль (джаз / поп / рок / электроника / латина / эмбиент);
  * тональность, темп, размер, гармоническая последовательность;
  * инструменты — случайные программы General MIDI (пианино, электропиано,
    гитара, струнные, духовые, орган, синты...);
  * барабаны — случайный паттерн (бочка/снейр/хэты/райд/открытый хэт, сбивки);
  * шум (белый/розовый), реверберация и хорус fluidsynth, случайная громкость;
  * прогон через mp3-кодек (как будто у тебя на входе реальный mp3).

Результат: data/*.npz (audio int16 + notes (N,3)) + data/index.json —
ровно то, что читает scripts/train.py.

Пример:
    python scripts/generate_data.py --out data --count 6000 --minutes 90
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pretty_midi
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SR  # noqa: E402

SOUNDFONT_CANDIDATES = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/FluidR3_GS.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
]

MAJOR = [0, 2, 4, 5, 7, 9, 11, 12]
MINOR = [0, 2, 3, 5, 7, 8, 10, 12]
DORIAN = [0, 2, 3, 5, 7, 9, 10, 12]

CHORDS = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "maj7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],
    "7": [0, 4, 7, 10],
    "dim": [0, 3, 6],
    "sus4": [0, 5, 7],
    "add9": [0, 4, 7, 14],
}

# программы General MIDI, из которых выбираются инструменты
GM_PIANO = [0, 1, 2, 3, 4, 5]
GM_GUITAR = [24, 25, 26, 27, 28, 29, 30, 31]
GM_STRINGS = [40, 41, 42, 44, 45, 46, 48, 49, 50]
GM_BRASS = [56, 57, 58, 60, 61, 62, 63]
GM_WIND = [64, 65, 66, 67, 68, 69, 71, 72, 73, 74]
GM_ORGAN = [16, 17, 18, 19, 20, 21]
GM_SYNTH = [80, 81, 82, 88, 89, 90, 94, 95]

STYLES = {
    "jazz": {
        "tempo": (110, 200),
        "swing": (0.58, 0.67),
        "scales": [MAJOR, MINOR, DORIAN],
        "progressions": [
            [(2, "m7"), (7, "7"), (0, "maj7"), (0, "maj7")],
            [(0, "maj7"), (5, "maj7"), (2, "m7"), (7, "7")],
            [(0, "m7"), (5, "7"), (0, "m7"), (7, "7")],
            [(5, "maj7"), (4, "m7"), (3, "maj7"), (2, "m7")],
        ],
        "bass": "walking",
        "melody_density": (0.55, 0.8),
        "drums": "jazz",
        "instruments": GM_PIANO + GM_GUITAR + GM_BRASS + GM_WIND + GM_ORGAN,
        "velocity": (55, 115),
    },
    "pop": {
        "tempo": (88, 132),
        "swing": None,
        "scales": [MAJOR, MINOR],
        "progressions": [
            [(0, "maj"), (4, "maj"), (5, "min"), (3, "maj")],
            [(5, "min"), (3, "maj"), (0, "maj"), (4, "maj")],
            [(0, "maj"), (4, "maj"), (0, "maj"), (5, "min")],
            [(0, "m7"), (5, "m7"), (3, "maj7"), (4, "7")],
        ],
        "bass": "root_eighths",
        "melody_density": (0.5, 0.75),
        "drums": "pop",
        "instruments": GM_PIANO + GM_GUITAR + GM_STRINGS + GM_SYNTH,
        "velocity": (70, 120),
    },
    "rock": {
        "tempo": (100, 160),
        "swing": None,
        "scales": [MINOR, DORIAN, MAJOR],
        "progressions": [
            [(0, "min"), (5, "min"), (2, "maj"), (6, "maj")],
            [(0, "min"), (3, "maj"), (4, "maj"), (0, "min")],
            [(0, "maj"), (4, "maj"), (5, "maj"), (3, "maj")],
        ],
        "bass": "root_eighths",
        "melody_density": (0.4, 0.65),
        "drums": "rock",
        "instruments": GM_GUITAR + GM_GUITAR + GM_ORGAN,
        "velocity": (80, 125),
    },
    "edm": {
        "tempo": (118, 140),
        "swing": None,
        "scales": [MINOR, DORIAN],
        "progressions": [
            [(0, "min"), (5, "min"), (2, "maj"), (6, "maj")],
            [(5, "min"), (4, "maj"), (0, "min"), (0, "min")],
        ],
        "bass": "offbeat",
        "melody_density": (0.7, 0.95),
        "drums": "edm",
        "instruments": GM_SYNTH + GM_SYNTH + GM_PIANO,
        "velocity": (75, 122),
    },
    "latin": {
        "tempo": (95, 135),
        "swing": None,
        "scales": [MINOR, DORIAN],
        "progressions": [
            [(0, "m7"), (3, "7"), (4, "maj7"), (0, "m7")],
            [(0, "m7"), (5, "7"), (0, "m7"), (4, "7")],
        ],
        "bass": "tumbao",
        "melody_density": (0.5, 0.8),
        "drums": "latin",
        "instruments": GM_PIANO + GM_GUITAR + GM_BRASS,
        "velocity": (60, 112),
    },
    "ambient": {
        "tempo": (60, 92),
        "swing": None,
        "scales": [MAJOR, MINOR, DORIAN],
        "progressions": [
            [(0, "maj7"), (5, "maj7"), (3, "maj7"), (4, "maj7")],
            [(0, "m7"), (4, "m7"), (0, "m7"), (5, "m7")],
        ],
        "bass": "long",
        "melody_density": (0.2, 0.45),
        "drums": "sparse",
        "instruments": GM_PIANO + GM_STRINGS + GM_SYNTH,
        "velocity": (45, 95),
    },
}

DRUM = {
    "kick": 36, "snare": 38, "clap": 39, "hat": 42, "open_hat": 46,
    "ride": 51, "crash": 49, "tom": 45, "rim": 37, "shaker": 70,
}


def drums_pattern(style, rng, bars=4):
    """Возвращает список (beat_offset_в_такте, нота, velocity) на 4 такта."""
    ev = []
    for bar in range(bars):
        b = bar * 4.0
        if style == "pop":
            for k in (0.0, 2.0):
                ev.append((b + k, DRUM["kick"], rng.randint(95, 120)))
            for s in (1.0, 3.0):
                ev.append((b + s, DRUM["snare"], rng.randint(90, 118)))
            for h in np.arange(0, 4, 0.5):
                ev.append((b + h, DRUM["hat"], rng.randint(45, 85)))
        elif style == "rock":
            for k in (0.0, 2.5):
                ev.append((b + k, DRUM["kick"], rng.randint(100, 125)))
            for s in (1.0, 3.0):
                ev.append((b + s, DRUM["snare"], rng.randint(100, 125)))
            for h in np.arange(0, 4, 0.5):
                ev.append((b + h, DRUM["hat"], rng.randint(50, 90)))
            if bar % 2 == 1 and rng.random() < 0.5:
                for t, p in ((3.5, DRUM["snare"]), (3.75, DRUM["tom"])):
                    ev.append((b + t, p, rng.randint(90, 120)))
        elif style == "jazz":
            for h in np.arange(0, 4, 0.5):
                ev.append((b + h, DRUM["ride"], rng.randint(45, 85)))
            for p in (1.0, 2.0, 3.0):
                if rng.random() < 0.55:
                    ev.append((b + p + rng.choice([0, 0.25]), DRUM["snare"], rng.randint(35, 75)))
            if rng.random() < 0.4:
                ev.append((b + rng.choice([0.0, 0.75]), DRUM["kick"], rng.randint(45, 70)))
            ev.append((b + 2.0, DRUM["hat"], rng.randint(50, 80)))
        elif style == "edm":
            for k in np.arange(0, 4, 1.0):
                ev.append((b + k, DRUM["kick"], rng.randint(110, 127)))
            for c in (1.0, 3.0):
                ev.append((b + c, DRUM["clap"], rng.randint(95, 120)))
            for h in np.arange(0.5, 4, 1.0):
                ev.append((b + h, DRUM["open_hat"], rng.randint(60, 95)))
        elif style == "latin":
            for h in np.arange(0, 4, 0.5):
                ev.append((b + h, DRUM["shaker"], rng.randint(35, 70)))
            for k in (0.0, 1.5, 2.5):
                ev.append((b + k, DRUM["kick"], rng.randint(80, 110)))
            for s in (1.0, 3.0):
                ev.append((b + s, DRUM["rim"], rng.randint(60, 95)))
            if rng.random() < 0.5:
                ev.append((b + 2.0, DRUM["tom"], rng.randint(70, 100)))
        else:  # sparse / ambient
            if bar % 2 == 0:
                ev.append((b, DRUM["kick"], rng.randint(50, 80)))
                ev.append((b + 2.0, DRUM["hat"], rng.randint(30, 60)))
    return ev


def build_midi(style_name, rng, bars, beats_per_bar):
    style = STYLES[style_name]
    scale = rng.choice(style["scales"])
    key = rng.choice([0, 2, 3, 5, 7, 8, 10])  # без самых «грязных» тональностей
    prog = rng.choice(style["progressions"])
    tempo = rng.uniform(*style["tempo"])
    seconds_per_beat = 60.0 / tempo
    swing = rng.uniform(*style["swing"]) if style["swing"] else None

    programs = rng.sample(style["instruments"], k=min(3, len(style["instruments"])))
    melody_p = programs[0]
    chord_p = programs[1] if len(programs) > 1 else programs[0]
    bass_p = rng.choice(GM_PIANO[:2] + GM_BRASS[:3] + [32, 33, 43])

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    mel = pretty_midi.Instrument(program=melody_p)
    cho = pretty_midi.Instrument(program=chord_p)
    bas = pretty_midi.Instrument(program=bass_p)
    drum = pretty_midi.Instrument(program=0, is_drum=True)

    lo, hi = style["velocity"]
    density = rng.uniform(*style["melody_density"])

    def beat_to_sec(beat):
        # свинг: чётные восьмые чуть длиннее
        b = beat
        if swing:
            whole = int(b)
            frac = b - whole
            if abs(frac - 0.5) < 1e-6:
                b = whole + swing
            elif abs(frac - 0.25) < 1e-6:
                b = whole + swing * 0.5
            elif abs(frac - 0.75) < 1e-6:
                b = whole + swing + (1 - swing) * 0.5
        return b * seconds_per_beat

    def add(inst, pitch, start_beat, dur_beats, vel):
        # свинг может сделать отображение beat->sec немонотонным,
        # поэтому конец ноты считаем «не раньше начала + 30 мс»
        s = beat_to_sec(start_beat)
        e = max(beat_to_sec(start_beat + dur_beats), s + 0.03)
        pitch = int(max(0, min(127, pitch)))
        inst.notes.append(pretty_midi.Note(velocity=int(vel), pitch=pitch, start=s, end=e))

    degree = rng.randint(0, 6)
    total_beats = bars * beats_per_bar

    for bar in range(bars):
        root_deg, chord_type = prog[bar % len(prog)]
        root_semi = key + scale[(root_deg) % 7] + 12 * ((root_deg) // 7)
        chord_intervals = CHORDS[chord_type]

        # --- аккорд: 1-2 удара в такте, случайные обращения
        hits = rng.choice([[0.0], [0.0, 2.0], [0.0, 1.5, 2.5], [0.0, 3.0]])
        for h in hits:
            start = bar * beats_per_bar + h
            dur = rng.uniform(0.4, 1.6) * beats_per_bar / 2
            base = 48 + (root_semi % 12)
            invert = rng.choice([0, 0, 12])
            for i in chord_intervals:
                pitch = int(base + i + invert)
                if 33 <= pitch <= 96:
                    add(cho, pitch, start, dur, rng.randint(lo, hi))

        # --- бас
        bass_mode = style["bass"]
        root_low = 28 + (root_semi % 12)
        if bass_mode == "walking":
            seq = [0, rng.choice(chord_intervals), 7, rng.choice([10, 11, 12])]
            for k, off in enumerate(seq):
                if rng.random() < 0.25:
                    continue
                add(bas, root_low + off, bar * beats_per_bar + k, 0.9, rng.randint(lo, hi))
        elif bass_mode == "root_eighths":
            for e in np.arange(0, 4, 1.0):
                add(bas, root_low, bar * beats_per_bar + e, 0.9, rng.randint(lo, hi))
        elif bass_mode == "offbeat":
            for e in np.arange(0.5, 4, 1.0):
                add(bas, root_low, bar * beats_per_bar + e, 0.45, rng.randint(lo, hi))
        elif bass_mode == "tumbao":
            for e in (0.0, 1.5, 2.5):
                add(bas, root_low, bar * beats_per_bar + e, 0.9, rng.randint(lo, hi))
        else:  # long
            add(bas, root_low, bar * beats_per_bar, beats_per_bar - 0.1, rng.randint(lo, hi))

        # --- мелодия: случайное блуждание по ладу
        t = 0.0
        while t < beats_per_bar:
            step = rng.choice([0.25, 0.5, 0.5, 1.0])
            if rng.random() > density:
                t += step
                continue
            degree = min(13, max(0, degree + rng.choice([-2, -1, -1, 0, 1, 1, 2])))
            pitch = int(60 + key % 12 + scale[degree % 8] + 12 * (degree // 8))
            pitch = min(98, max(45, pitch))
            dur = step * rng.uniform(0.6, 1.0)
            if rng.random() < 0.8:  # иногда короткий акцент
                add(mel, pitch, bar * beats_per_bar + t, dur, rng.randint(lo, hi))
            t += step

    # --- барабаны
    if rng.random() < 0.85:
        for beat, pitch, vel in drums_pattern(style["drums"], rng, bars):
            add(drum, pitch, beat, 0.1, vel)

    for inst in (mel, cho, bas):
        if inst.notes:
            pm.instruments.append(inst)
    if drum.notes:
        pm.instruments.append(drum)
    return pm


def render(pm, wav_path, sf2, rng):
    midi_path = wav_path + ".mid"
    pm.write(midi_path)
    cmd = [
        "fluidsynth", "-ni", "-F", wav_path, "-r", str(SR),
        "-g", f"{rng.uniform(0.6, 1.1):.2f}",
        "-o", "synth.reverb.active=1",
        "-o", f"synth.reverb.room-size={rng.uniform(0.3, 0.95):.2f}",
        "-o", f"synth.reverb.level={rng.uniform(0.2, 0.9):.2f}",
        "-o", f"synth.chorus.active={1 if rng.random() < 0.4 else 0}",
        sf2, midi_path,
    ]
    subprocess.run(cmd, check=True, timeout=180,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(midi_path)
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    assert sr == SR, f"fluidsynth выдал {sr} Гц"
    return audio.mean(axis=1)


def degrade(audio, rng, tmpdir, mp3_prob, noise_prob):
    """Шум, громкость, mp3-кодек — чтобы модель не привыкала к идеальному звуку."""
    audio = audio.copy()

    if rng.random() < noise_prob:
        snr_db = rng.uniform(12, 40)
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
        noise = np.random.default_rng(rng.randrange(2 ** 31)).normal(0, 1, len(audio)).astype(np.float32)
        # немного «розовее» белого шума
        noise = np.convolve(noise, np.ones(3, np.float32) / 3, mode="same")
        noise_rms = float(np.sqrt(np.mean(noise ** 2)) + 1e-12)
        audio = audio + noise * (rms / noise_rms) * (10 ** (-snr_db / 20))

    gain = rng.uniform(0.25, 1.15)
    audio = np.clip(audio * gain, -1.0, 1.0)

    if rng.random() < mp3_prob and shutil.which("ffmpeg"):
        wav_in = os.path.join(tmpdir, "in.wav")
        mp3 = os.path.join(tmpdir, "tmp.mp3")
        wav_out = os.path.join(tmpdir, "out.wav")
        sf.write(wav_in, audio, SR)
        bitrate = rng.choice(["64k", "96k", "128k", "192k"])
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_in,
                        "-c:a", "libmp3lame", "-b:a", bitrate, mp3], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp3,
                        "-ac", "1", "-ar", str(SR), wav_out], check=True)
        audio, _ = sf.read(wav_out, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio


def make_track(job):
    idx, seed, out_path, sf2, min_s, max_s, mp3_prob, noise_prob = job
    rng = random.Random(seed)
    np.random.seed(seed % (2 ** 31))

    style_name = rng.choice(list(STYLES.keys()))
    want = rng.uniform(min_s, max_s)
    tempo_guess = rng.uniform(80, 140)
    bars = max(2, int(np.ceil(want / (4 * 60 / tempo_guess))))

    pm = build_midi(style_name, rng, bars, 4)
    notes = np.array(
        [(n.pitch, n.start, n.end) for inst in pm.instruments if not inst.is_drum for n in inst.notes],
        dtype=np.float32,
    ).reshape(-1, 3)
    if len(notes) < 4:
        return None

    tmpdir = out_path + ".tmp"
    os.makedirs(tmpdir, exist_ok=True)
    wav = os.path.join(tmpdir, "render.wav")
    try:
        audio = render(pm, wav, sf2, rng)
        audio = degrade(audio, rng, tmpdir, mp3_prob, noise_prob)

        # обрезаем до нужной длины
        keep = min(len(audio), int(want * SR))
        audio = audio[:keep]
        if len(audio) < int(min_s * SR * 0.8):
            return None
        notes = notes[notes[:, 1] < keep / SR]
        if len(notes) < 4:
            return None

        np.savez(out_path, audio=(audio * 32767).astype(np.int16), notes=notes)
        return (out_path, style_name, len(audio) / SR, len(notes))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--count", type=int, default=6000)
    ap.add_argument("--minutes", type=float, default=90, help="бюджет времени на генерацию")
    ap.add_argument("--min-seconds", type=float, default=6)
    ap.add_argument("--max-seconds", type=float, default=14)
    ap.add_argument("--noise-prob", type=float, default=0.6)
    ap.add_argument("--mp3-prob", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0, help="0 = по числу ядер")
    ap.add_argument("--soundfont", default=None)
    ap.add_argument("--val-ratio", type=float, default=0.06)
    args = ap.parse_args()

    sf2 = args.soundfont or next((p for p in SOUNDFONT_CANDIDATES if os.path.exists(p)), None)
    if not sf2:
        raise SystemExit("Нет саундфонта: sudo apt-get install -y fluid-soundfont-gm")
    if not shutil.which("fluidsynth"):
        raise SystemExit("Нет fluidsynth: sudo apt-get install -y fluidsynth")

    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, "index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
    else:
        index = {"train": [], "validation": []}

    have = len(index["train"]) + len(index["validation"])
    todo = max(0, args.count - have)
    workers = args.workers or max(1, os.cpu_count() or 1)
    deadline = time.time() + args.minutes * 60

    print(f"[gen] стилей: {len(STYLES)} ({', '.join(STYLES)}), воркеров: {workers}")
    print(f"[gen] уже есть: {have}, генерирую: {todo} (бюджет {args.minutes:.0f} мин)")

    rng = random.Random(args.seed + have)
    jobs = []
    made_val = len(index["validation"])
    want_val = max(20, int(args.count * args.val_ratio))
    for k in range(todo):
        idx = have + k
        split = "validation" if made_val < want_val and (idx % 7 == 0) else "train"
        if split == "validation":
            made_val += 1
        jobs.append((
            idx, rng.randrange(1 << 30), os.path.join(args.out, f"{split}_{idx:06d}.npz"),
            sf2, args.min_seconds, args.max_seconds, args.mp3_prob, args.noise_prob,
        ))

    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(make_track, j): j for j in jobs}
        for fut in as_completed(futures):
            if time.time() > deadline:
                for f in futures:
                    f.cancel()
                print("[gen] бюджет времени исчерпан — останавливаюсь")
                break
            j = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"[gen] пропуск {j[2]}: {e}")
                continue
            if not res:
                continue
            path, style_name, dur, n_notes = res
            split = "validation" if os.path.basename(path).startswith("validation") else "train"
            index[split].append(path)
            done += 1
            if done % 200 == 0:
                speed = done / max(1e-6, time.time() - t0)
                left = (todo - done) / max(1e-6, speed) / 60
                print(f"[gen] {done}/{todo} ({speed:.1f} трек/с, осталось ~{left:.0f} мин) "
                      f"последний: {style_name} {dur:.1f}с")

    with open(index_path, "w") as f:
        json.dump(index, f)

    total_min = (time.time() - t0) / 60
    size = sum(os.path.getsize(p) for p in index["train"] + index["validation"]) / 1e9
    with open(os.path.join(args.out, "gen_stats.json"), "w") as f:
        json.dump({"count": len(index["train"]) + len(index["validation"]),
                   "train": len(index["train"]), "validation": len(index["validation"]),
                   "minutes": round(total_min, 1), "gb": round(size, 2),
                   "per_clip_sec": round(total_min * 60 / max(1, done), 2)}, f, indent=2)
    print(f"[gen] готово: {len(index['train'])} train / {len(index['validation'])} val, "
          f"{total_min:.1f} мин, {size:.2f} ГБ")


if __name__ == "__main__":
    main()
