import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from pydub import AudioSegment
from tqdm import tqdm
from transformers import HfArgumentParser


@dataclass
class ProcessArguments:
    data_folder: Optional[str] = field(default=None)
    output_folder: Optional[str] = field(default=None)
    split: Optional[str] = field(default=None)
    max_seconds: Optional[int] = field(default=3)
    overlap_seconds: Optional[float] = field(default=0.5)
    peak_norm_db: Optional[float] = field(default=-5.0)


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def chunk_data(data, max_length, overlap_width):
    data_length = data.shape[1]
    if data_length <= max_length:
        return [(data, data_length)]
    chunks = []
    cur_start = 0
    while cur_start < data_length:
        end = min(cur_start + max_length, data_length)
        chunks.append((data[:, cur_start:end], end - cur_start))
        if end == data_length:
            break
        cur_start = end - overlap_width
    return chunks


def parse_files(files, args):
    max_seconds = args.max_seconds
    peak_norm_db = args.peak_norm_db

    lengths = []
    legit_files = []
    legit_text = []
    legit_speech = 0
    save_folder = f"{args.output_folder}/{args.split}"
    os.makedirs(save_folder, exist_ok=True)
    for file in tqdm(files, total=len(files)):
        if '_normalized' in file.stem or "chunk" in file.stem:
            continue

        text_file = f"{file.parent}/{file.stem}.normalized.txt"
        try:
            data, sample_hz = torchaudio.load(file)
        except:
            print(f'file {file} is broken')
            continue

        # we first load the audio and separate it into consecutive segments of max_seconds
        max_length = int(sample_hz * max_seconds)
        if data.shape[1] < max_length:
            numpy_data = data.numpy()
            sound_segment = AudioSegment(numpy_data.tobytes(),
                                         frame_rate=sample_hz,
                                         sample_width=numpy_data.dtype.itemsize,
                                         channels=1)
            dbfs_value = sound_segment.dBFS
            db_diff = peak_norm_db - dbfs_value

            linear_gain = 10 ** (db_diff / 20)
            direct_norm_sound = torch.clamp(data * linear_gain, -1, 1)

            normalized_file = f"{save_folder}/{file.stem}_norm.wav"
            torchaudio.save(normalized_file, direct_norm_sound, sample_hz)

            with open(text_file, "r") as f:
                text_data = f.read().replace("\n", " ").strip()
            text_file = f"{save_folder}/{file.stem}_norm.txt"
            with open(text_file, "w") as wf:
                print(text_data, file=wf)

            legit_speech += 1
            lengths.append(data.shape[1])
            legit_files.append(normalized_file)
            legit_text.append(text_file)
    return legit_files, lengths, legit_text


def save(files, lengths, all_text, output_dir, split):
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{split}.txt"
    with open(output_file, 'w') as f:
        for file, length, text in zip(files, lengths, all_text):
            print(f"{file}\t{length}\t{text}", file=f)


def main(args):
    data_folder = args.data_folder
    output_folder = args.output_folder
    split = args.split
    path = Path(data_folder)
    files = []
    for file in tqdm(path.glob('**/*.wav')):
        if file.is_file() and 'normalized' not in file.stem and 'chunk' not in file.stem:
            files.append(file)

    files, lengths, all_text = parse_files(files, args)
    save(files, lengths, all_text, output_folder, split)


if __name__ == "__main__":
    args = HfArgumentParser(ProcessArguments).parse_args_into_dataclasses()[0]
    print(args)
    main(args)
