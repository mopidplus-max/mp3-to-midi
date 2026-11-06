#!/usr/bin/env python3
"""audio (mp3/wav/m4a/...) -> MIDI.

Два режима:
  * без --model-path — официальная предобученная Basic Pitch (качается вместе с pip-пакетом).
    Это то, что нужно, если цель — просто получить MIDI из mp3.
  * с --model-path out/saved_model — модель, которую обучил scripts/train.py.

Пример:
    python scripts/infer.py song.mp3 -o midi/
    python scripts/infer.py song.mp3 -o midi/ --model-path out/saved_model
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

AUDIO_EXT = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".mp4", ".webm")


def to_wav(path, workdir):
    """Приводит любой файл к wav 22 050 Гц моно, чтобы не зависеть от кодеков."""
    out = os.path.join(workdir, "input.wav")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", path,
               "-ac", "1", "-ar", "22050", out]
        subprocess.run(cmd, check=True)
        return out
    if path.lower().endswith(".wav"):
        return path
    raise SystemExit("Нужен ffmpeg для декодирования mp3. Установи: apt-get install -y ffmpeg")


def collect_inputs(items):
    files = []
    for item in items:
        if os.path.isdir(item):
            for ext in AUDIO_EXT:
                files.extend(glob.glob(os.path.join(item, "**", "*" + ext), recursive=True))
        elif os.path.isfile(item):
            files.append(item)
        else:
            print(f"[infer] пропуск (не файл): {item}")
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="файлы или папки с аудио")
    ap.add_argument("-o", "--out", default="midi", help="куда писать .mid")
    ap.add_argument("--model-path", default=None,
                    help="своя обученная модель (папка saved_model). По умолчанию — предобученная")
    ap.add_argument("--onset-threshold", type=float, default=0.5)
    ap.add_argument("--frame-threshold", type=float, default=0.3)
    ap.add_argument("--min-note-ms", type=float, default=58.0,
                    help="минимальная длина ноты в мс (меньше — меньше мусора)")
    args = ap.parse_args()

    from basic_pitch.inference import predict

    files = collect_inputs(args.inputs)
    if not files:
        print("[infer] аудиофайлов не найдено — нечего транскрибировать. "
              "Положи mp3 в input/ или передай ссылку в audio_url.")
        return

    os.makedirs(args.out, exist_ok=True)
    model = args.model_path or None

    tmpdir = tempfile.mkdtemp()
    ok = 0
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            wav = to_wav(path, tmpdir)
            kwargs = dict(
                onset_threshold=args.onset_threshold,
                frame_threshold=args.frame_threshold,
                minimum_note_length=args.min_note_ms,
            )
            if model:
                _, midi_data, _ = predict(wav, model_or_model_path=model, **kwargs)
            else:
                _, midi_data, _ = predict(wav, **kwargs)

            out_path = os.path.join(args.out, name + ".mid")
            midi_data.write(out_path)
            n_notes = sum(len(i.notes) for i in midi_data.instruments)
            print(f"[infer] {name}: {n_notes} нот -> {out_path}")
            ok += 1
        except Exception as e:
            print(f"[infer] ошибка на {path}: {e}")

    print(f"[infer] готово: {ok}/{len(files)} файлов")
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
