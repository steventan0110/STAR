#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path
from itertools import groupby
import soundfile as sf

import torch
import yaml
import torchaudio
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser

from fairseq.data.audio.audio_utils import get_waveform, convert_waveform


log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]

@dataclass
class ProcessArguments:
    data_folder: Optional[str] = field(default=None)
    output_folder: Optional[str] = field(default=None)
    split: Optional[str] = field(default=None)
    lang: Optional[str] = field(default="de")


def main(args):
    lang = args.lang
    if args.split == "train" or args.split == "dev":
        data_dir= f"{args.data_folder}/{args.split}"
        txt_yaml = f"{data_dir}/txt/{args.split}.yaml"
        src_data_file = f"{data_dir}/txt/{args.split}.en"
        tgt_data_file  = f"{data_dir}/txt/{args.split}.{lang}"
    else:
        data_dir = f"{args.data_folder}/tst-COMMON" # special name for test
        txt_yaml = f"{data_dir}/txt/tst-COMMON.yaml"
        src_data_file  = f"{data_dir}/txt/tst-COMMON.en"
        tgt_data_file  = f"{data_dir}/txt/tst-COMMON.{lang}"
    wav_root = f"{data_dir}/wav"

    # load yaml file for duration
    with open(txt_yaml) as f:
        segments = yaml.load(f, Loader=yaml.BaseLoader)

    # prepare
    with open(src_data_file, "r") as f:
        src_data = [r.strip() for r in f]
        assert len(src_data) == len(segments)
    with open(tgt_data_file, "r") as f:
        tgt_data = [r.strip() for r in f]
        assert len(tgt_data) == len(segments)
    for i in range(len(segments)):
        segments[i]["en"] = src_data[i]
        segments[i][lang] = tgt_data[i]
    # now we process the audio data and align the txt
    num_files = 0
    file2save = f"{args.output_folder}/{args.split}.txt"
    file_hanlde = open(file2save, "w")
    for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
        wav_path = Path(f"{wav_root}/{wav_filename}")
        sample_rate = sf.info(wav_path.as_posix()).samplerate
        seg_group = sorted(_seg_group, key=lambda x: x["offset"])
        for i, segment in enumerate(seg_group):
            offset = int(float(segment["offset"]) * sample_rate)
            n_frames = int(float(segment["duration"]) * sample_rate)
            # if n_frames < sample_rate * 1:
            #     continue
            if args.split != "test" and n_frames > sample_rate * 15:
                continue
            _id = f"{wav_path.stem}_{i}"
            waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
            waveform = torch.from_numpy(waveform)
            filename = wav_filename.replace(".wav", "")
            wav2save = f"{args.output_folder}/{args.split}/{filename}_seg{i}.wav"

            # start saving audio and aligned txt
            src_line = segment["en"]
            tgt_line = segment[lang]
            # line2write = f"{wav2save}\t{n_frames}\t{src_line}\t{tgt_line}"
            # we do not use English side info for simulst
            line2write = f"{wav2save}\t{n_frames}\t{tgt_line}"
            print(line2write, file=file_hanlde)

            os.makedirs(f"{args.output_folder}/{args.split}", exist_ok=True)
            num_files += 1
            torchaudio.save(wav2save, waveform, sample_rate)
    print(f"Num of cleaned chunked audio files: {num_files}")
    print(f"Total number of text lines: {len(src_data)}")
    file_hanlde.close()



if __name__ == "__main__":
    args = HfArgumentParser(ProcessArguments).parse_args_into_dataclasses()[0]
    print(args)
    main(args)