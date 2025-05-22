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
    num_filtered = 0

    folder2txt = {}
    # preprocessing folders, obtain the txt file for each folder
    for file in tqdm(files, total=len(files)):
        if not file.parent in folder2txt:
            txt_ids = file.stem.split("-")
            assert len(txt_ids) == 3
            txt_id = "-".join(txt_ids[:2])
            txt_file = f"{file.parent}/{txt_id}.trans.txt"
            folder2txt[file.parent] = txt_file
    # read the txt file and pair the txt with audio id
    sample_hz = 16000
    max_length = int(sample_hz * max_seconds)
    for folder_path, txt_file in tqdm(folder2txt.items(), total=len(folder2txt)):
        with open(txt_file, "r") as f:
            for line in f:
                audio_id, txt = line.rstrip().split(" ", 1)
                audio_path = f"{folder_path}/{audio_id}.flac"
                try:
                    data, _ = torchaudio.load(audio_path)
                except:
                    print(f'file {audio_path} is broken')
                    continue

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

                    normalized_file = f"{save_folder}/{audio_id}_norm.wav"
                    torchaudio.save(normalized_file, direct_norm_sound, sample_hz)
                    text_file = f"{save_folder}/{audio_id}_norm.txt"
                    with open(text_file, "w") as wf:
                        txt = txt.lower()
                        print(txt, file=wf)

                    legit_speech += 1
                    lengths.append(data.shape[1])
                    legit_files.append(normalized_file)
                    legit_text.append(text_file)
                else:
                    num_filtered += 1

    print(f"Filtered {num_filtered} files; Kept {legit_speech} files.")
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

    for file in tqdm(path.glob('**/*.flac')):
        if file.is_file():
            files.append(file)
    files, lengths, all_text = parse_files(files, args)
    save(files, lengths, all_text, output_folder, split)


if __name__ == "__main__":
    args = HfArgumentParser(ProcessArguments).parse_args_into_dataclasses()[0]
    print(args)
    main(args)
