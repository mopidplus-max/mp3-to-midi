"""Общие функции: MIDI -> целевые матрицы (onset / note / contour) и загрузка данных.

Раскладка таргетов повторяет Basic Pitch (Spotify):
  * аудио 22 050 Гц, моно, окно ~2 секунды (43 844 сэмпла)
  * сетка кадров ~86 кадров/сек (22050 // 256), окно = 172 кадра
  * notes/onsets: (172, 88)      — 88 клавиш пианино, начиная с MIDI 21 (A0)
  * contours:     (172, 264)     — те же 88 клавиш, но 3 бина на полутон
"""

import numpy as np

from basic_pitch.constants import (
    ANNOTATIONS_FPS,
    ANNOTATIONS_N_SEMITONES,
    CONTOURS_BINS_PER_SEMITONE,
    NOTES_BINS_PER_SEMITONE,
    AUDIO_N_SAMPLES,
    AUDIO_SAMPLE_RATE,
    ANNOT_N_FRAMES,
)

FPS = ANNOTATIONS_FPS                      # 86 кадров в секунду
MIDI_LOWEST = 21                           # A0 — первая клавиша пианино
N_NOTE_BINS = ANNOTATIONS_N_SEMITONES * NOTES_BINS_PER_SEMITONE      # 88
N_CONTOUR_BINS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE  # 264
WINDOW_FRAMES = ANNOT_N_FRAMES             # 172
WINDOW_SAMPLES = AUDIO_N_SAMPLES           # 43844
SR = AUDIO_SAMPLE_RATE                     # 22050


def notes_to_targets(notes, window_start, n_frames=WINDOW_FRAMES):
    """notes: массив (N, 3) -> [midi_pitch, start_sec, end_sec].

    Возвращает onset, note, contour — всё float32.
    """
    onset = np.zeros((n_frames, N_NOTE_BINS), dtype=np.float32)
    note = np.zeros((n_frames, N_NOTE_BINS), dtype=np.float32)
    contour = np.zeros((n_frames, N_CONTOUR_BINS), dtype=np.float32)

    for pitch, start, end in notes:
        b = int(round(float(pitch))) - MIDI_LOWEST
        if b < 0 or b >= N_NOTE_BINS:
            continue

        i0 = int(np.ceil((float(start) - window_start) * FPS))
        i1 = int(np.ceil((float(end) - window_start) * FPS))

        a, z = max(i0, 0), min(i1, n_frames)
        if z > a:
            note[a:z, b] = 1.0
            contour[a:z, CONTOURS_BINS_PER_SEMITONE * b] = 1.0

        if 0 <= i0 < n_frames and i1 > i0:
            onset[i0, b] = 1.0

    return onset, note, contour


def pick_window(notes, duration, rng, min_notes=1, tries=8):
    """Случайное начало окна, желательно с нотами внутри."""
    win = WINDOW_SAMPLES / SR
    max_start = max(0.0, duration - win)
    best = float(rng.uniform(0.0, max_start)) if max_start > 0 else 0.0
    if len(notes) == 0:
        return best

    starts = np.asarray(notes)[:, 1]
    for _ in range(tries):
        cand = float(rng.uniform(0.0, max_start)) if max_start > 0 else 0.0
        inside = np.sum((starts >= cand) & (starts < cand + win))
        if inside >= min_notes:
            return cand
        best = cand
    return best


def crop_audio(audio, start_sec):
    """Вырезает окно фиксированной длины, при необходимости добивает нулями."""
    i = int(round(start_sec * SR))
    win = audio[i:i + WINDOW_SAMPLES]
    if len(win) < WINDOW_SAMPLES:
        win = np.pad(win, (0, WINDOW_SAMPLES - len(win)))
    return win.astype(np.float32)


def load_track(path):
    """Читает .npz, который делает scripts/prepare_data.py.

    Возвращает (audio float32 mono в [-1, 1], notes (N,3) float32).
    """
    with np.load(path) as z:
        audio = z["audio"].astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        notes = z["notes"].astype(np.float32).reshape(-1, 3)
    return audio, notes
