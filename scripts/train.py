#!/usr/bin/env python3
"""Обучение audio -> notes модели (архитектура Basic Pitch) с нуля.

Особенности, заточенные под GitHub-раннер:
  * жёсткий лимит по времени (--minutes): по его истечении обучение
    аккуратно останавливается и чекпойнт сохраняется, а не теряется
    вместе с убитым джобом;
  * чекпойнт и экспортированная модель кладутся в --out, откуда workflow
    забирает их артефактом;
  * можно продолжить обучение из прошлого чекпойнта (--init-from).

Пример:
    python scripts/train.py --data data --out out --minutes 300 --batch-size 8
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

try:  # чтобы прогресс было видно в логах GitHub сразу, а не в конце
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    N_CONTOUR_BINS,
    N_NOTE_BINS,
    SR,
    WINDOW_FRAMES,
    WINDOW_SAMPLES,
    crop_audio,
    load_track,
    notes_to_targets,
    pick_window,
)

from basic_pitch import models  # noqa: E402

AUTOTUNE = tf.data.AUTOTUNE


def read_index(data_dir):
    with open(os.path.join(data_dir, "index.json")) as f:
        index = json.load(f)
    for split in ("train", "validation"):
        if not index.get(split):
            raise SystemExit(f"В index.json нет {split}-треков. Сначала prepare_data.py")
    return index


def _load_window(path, train, seed):
    """numpy-часть: читает трек и возвращает один обучающий пример."""
    import random as _random

    name = path.numpy().decode() if hasattr(path, "numpy") else str(path)
    is_train = bool(train)
    seed_val = int(seed)
    rng = _random.Random((seed_val + (abs(hash(name)) % 100000)) if is_train else seed_val)
    audio, notes = load_track(name)

    duration = len(audio) / SR
    if is_train:
        start = pick_window(notes, duration, rng)
    else:
        start = max(0.0, duration / 2.0 - (WINDOW_SAMPLES / SR) / 2.0)

    onset, note, contour = notes_to_targets(notes, start, WINDOW_FRAMES)

    return (
        crop_audio(audio, start),
        onset.astype(np.float32),
        note.astype(np.float32),
        contour.astype(np.float32),
    )


def make_dataset(paths, batch_size, train, seed, parallel=2):
    ds = tf.data.Dataset.from_tensor_slices(np.array(paths))
    if train:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)
    ds = ds.repeat()

    def _map(path):
        a, o, n, c = tf.py_function(_load_window, [path, train, seed],
                                    [tf.float32, tf.float32, tf.float32, tf.float32])
        a = tf.ensure_shape(a, [WINDOW_SAMPLES])
        o = tf.ensure_shape(o, [WINDOW_FRAMES, N_NOTE_BINS])
        n = tf.ensure_shape(n, [WINDOW_FRAMES, N_NOTE_BINS])
        c = tf.ensure_shape(c, [WINDOW_FRAMES, N_CONTOUR_BINS])
        return tf.expand_dims(a, -1), {"onset": o, "note": n, "contour": c}

    ds = ds.map(_map, num_parallel_calls=parallel)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(AUTOTUNE)


class TimeBudget(tf.keras.callbacks.Callback):
    """Останавливает обучение до того, как раннер прибьёт джоб."""

    def __init__(self, minutes, reserve_seconds=180):
        super().__init__()
        self.limit = minutes * 60 - reserve_seconds
        self.start = None

    def on_train_begin(self, logs=None):
        self.start = time.time()
        print(f"[train] бюджет: {self.limit/60:.1f} мин чистого обучения")

    def on_train_batch_end(self, batch, logs=None):
        elapsed = time.time() - self.start
        if elapsed >= self.limit:
            print(f"[train] время вышло ({elapsed/60:.1f} мин) — останавливаюсь и сохраняюсь")
            self.model.stop_training = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--minutes", type=float, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--validation-steps", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--no-contours", action="store_true",
                    help="учить без контуров (чуть быстрее, но хуже тремоло/вибрато)")
    ap.add_argument("--init-from", default=None,
                    help="продолжить из чекпойнта прошлого запуска (папка с weights.index)")
    ap.add_argument("--threads", type=int, default=0, help="0 = авто")
    args = ap.parse_args()

    if args.threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, args.threads // 2))

    os.makedirs(args.out, exist_ok=True)
    index = read_index(args.data)
    print(f"[train] треков: {len(index['train'])} train / {len(index['validation'])} val")

    model = models.model(no_contours=args.no_contours)
    if args.init_from:
        print(f"[train] продолжаю из {args.init_from}")
        status = model.load_weights(args.init_from)
        if hasattr(status, "expect_partial"):
            status.expect_partial()

    train_ds = make_dataset(index["train"], args.batch_size, True, seed=1)
    val_ds = make_dataset(index["validation"], args.batch_size, False, seed=7)

    model.compile(
        loss=models.loss(),
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
    )
    model.summary(print_fn=lambda s, **k: print(s))

    ckpt_dir = os.path.join(args.out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "weights")

    callbacks = [
        TimeBudget(args.minutes),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, verbose=1,
                                             min_lr=1e-5),
        tf.keras.callbacks.CSVLogger(os.path.join(args.out, "training_log.csv")),
    ]

    t0 = time.time()
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=val_ds,
        validation_steps=args.validation_steps,
        callbacks=callbacks,
        verbose=2,
    )

    model.save_weights(ckpt_path)
    print(f"[train] чекпойнт: {ckpt_path}")

    export_dir = os.path.join(args.out, "saved_model")
    tf.saved_model.save(model, export_dir)
    print(f"[train] экспортировано: {export_dir}")

    stats = {
        "minutes_used": round((time.time() - t0) / 60, 2),
        "steps": len(history.history.get("loss", [])),
        "final_loss": history.history.get("loss", [None])[-1],
        "final_val_loss": (history.history.get("val_loss") or [None])[-1],
        "no_contours": args.no_contours,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("[train] итог:", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
