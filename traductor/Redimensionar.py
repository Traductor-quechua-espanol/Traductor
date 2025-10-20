import os
import argparse
import subprocess
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm


def run_ffmpeg(in_path, out_path, sample_rate=16000):
    """
    Convierte audio a WAV mono 16k usando ffmpeg
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-ar", str(sample_rate),
        "-ac", "1",
        str(out_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def convert_all(data_root, out_root):
    """
    Convierte todos los .mp3 a wav 16k
    """
    data_root = Path(data_root)
    out_root = Path(out_root)
    files = list(data_root.rglob("*.mp3"))

    for f in tqdm(files, desc="Convirtiendo a WAV 16k"):
        rel = f.relative_to(data_root)
        out_path = out_root / rel.with_suffix(".wav")
        run_ffmpeg(f, out_path)


def trim_silence(in_path, out_path, top_db=30):
    """
    Recorta silencios usando librosa.effects.split
    """
    y, sr = librosa.load(in_path, sr=None)
    intervals = librosa.effects.split(y, top_db=top_db)

    if len(intervals) == 0:
        y_trimmed = y
    else:
        y_trimmed = np.concatenate([y[start:end] for start, end in intervals])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, y_trimmed, sr)


def normalize_audio(in_path, out_path, target_db=-20.0):
    """
    Normaliza audio a un nivel RMS aproximado
    """
    y, sr = librosa.load(in_path, sr=None)
    rms = np.sqrt(np.mean(y**2))
    if rms > 0:
        scalar = 10**(target_db/20) / rms
        y = y * scalar

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, y, sr)


def run_pipeline(data_root, out_root):
    """
    Ejecuta todo: convertir -> recortar silencio -> normalizar
    """
    temp_root = Path(out_root) / "wav16"
    trimmed_root = Path(out_root) / "trimmed"
    norm_root = Path(out_root) / "normalized"

    # 1. Convertir a wav 16k
    convert_all(data_root, temp_root)

    # 2. Recortar silencios
    for f in tqdm(list(temp_root.rglob("*.wav")), desc="Recortando silencios"):
        rel = f.relative_to(temp_root)
        out_path = trimmed_root / rel
        trim_silence(f, out_path)

    # 3. Normalizar
    for f in tqdm(list(trimmed_root.rglob("*.wav")), desc="Normalizando"):
        rel = f.relative_to(trimmed_root)
        out_path = norm_root / rel
        normalize_audio(f, out_path)

    print("✅ Pipeline completo. Carpeta final:", norm_root)


if __name__ == "_main_":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["convert", "trim", "normalize", "all"])
    parser.add_argument("--data_root", required=True, help="Carpeta con los audios originales (mp3)")
    parser.add_argument("--out_root", required=True, help="Carpeta donde guardar resultados")
    args = parser.parse_args()

    if args.action == "convert":
        convert_all(args.data_root, args.out_root)
    elif args.action == "trim":
        for f in Path(args.data_root).rglob("*.wav"):
            rel = f.relative_to(args.data_root)
            out_path = Path(args.out_root) / rel
            trim_silence(f, out_path)
    elif args.action == "normalize":
        for f in Path(args.data_root).rglob("*.wav"):
            rel = f.relative_to(args.data_root)
            out_path = Path(args.out_root) / rel
            normalize_audio(f, out_path)
    elif args.action == "all":
        run_pipeline(args.data_root, args.out_root)