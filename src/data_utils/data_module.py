from typing import Optional, List
import re

import torch
import torchaudio
from einops import rearrange, reduce
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset, Sampler
from torchaudio.functional import resample

IGNORE_INDEX = -100

class SpeechSampler(Sampler):
    def __init__(
            self,
            batch_size: int,
            lengths: Optional[List[int]] = None,
            generator=None,
    ):
        super().__init__(None)
        self.batch_size = batch_size
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.tolist()
        self.lengths = lengths
        self.generator = generator

    @staticmethod
    def get_length_grouped_indices(
            lengths, batch_size, mega_batch_mult=None, generator=None
    ):
        """
        Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
        lengths. To do this, the indices are:

        - randomly permuted
        - grouped in mega-batches of size `mega_batch_mult * batch_size`
        - sorted by length in each mega-batch

        The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
        maximum length placed first, so that an OOM happens sooner rather than later.
        """
        # Default for mega_batch_mult: 50 or the number to get 4 megabatches, whichever is smaller.
        if mega_batch_mult is None:
            mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
            # Just in case, for tiny datasets
            if mega_batch_mult == 0:
                mega_batch_mult = 1

        # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
        indices = torch.randperm(len(lengths), generator=generator)
        megabatch_size = mega_batch_mult * batch_size
        megabatches = [
            indices[i: i + megabatch_size].tolist()
            for i in range(0, len(lengths), megabatch_size)
        ]
        megabatches = [
            sorted(megabatch, key=lambda i: lengths[i], reverse=True)
            for megabatch in megabatches
        ]

        # The rest is to get the biggest batch first.
        # Since each megabatch is sorted by descending length, the longest element is the first
        megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
        max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
        # Switch to put the longest element in first position
        megabatches[0][0], megabatches[max_idx][0] = (
            megabatches[max_idx][0],
            megabatches[0][0],
        )

        return [i for megabatch in megabatches for i in megabatch]

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = self.get_length_grouped_indices(
            self.lengths, self.batch_size, generator=self.generator
        )
        return iter(indices)


class SpeechDatatset(Dataset):
    def __init__(self, data_file, target_sample_hz=16000, max_seconds=3, overlap_audio=False, limit=None,
                 return_all=False):
        super(SpeechDatatset).__init__()
        self.data_file = data_file
        self.target_sample_hz = target_sample_hz
        self.max_seconds = max_seconds
        self.max_length = self.target_sample_hz * max_seconds
        self.audios_files = []
        self.lengths = []
        self.return_all = return_all
        # TODO : implement a version that chunk audio into overlapping segments
        self.overlap_audio = overlap_audio

        with open(data_file, 'r') as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                if len(line) > 0:
                    filepath, length = line.strip().split('\t')
                    self.audios_files.append(filepath)
                    self.lengths.append(int(length))
        print(f"Loaded {len(self.audios_files)} audio files")

    def __getitem__(self, index):
        file = self.audios_files[index]
        data, sample_rate = torchaudio.load(file)
        if data.shape[0] > 1:
            # the audio has more than 1 channel, convert to mono
            data = reduce(data, 'c ... -> 1 ...', 'mean')

        data = resample(data, sample_rate, self.target_sample_hz)
        audio_length = data.size(1)
        if self.return_all:
            # for test time inference on the whole audio
            return data, audio_length

        if audio_length > self.max_length:
            max_start = audio_length - self.max_length
            # randomly retrieve a segment of the audio that has max length
            start = torch.randint(0, max_start, (1,))
            data = data[:, start:start + self.max_length]
            audio_length = self.max_length

        # remove the first dimension as it's always 1
        data = rearrange(data, '1 ... -> ...')
        # TODO: could support multiple target samplze hz, but for now, let's just stay with one target
        return data, audio_length

    def __len__(self):
        return len(self.audios_files)

    def get_lengths(self):
        return self.lengths


class SpeechTextDatatset(Dataset):
    def __init__(
            self, data_file, target_sample_hz=16000, max_seconds=3, overlap_audio=False, limit=None, return_all=False, normalize=False,
             lm_tokenizer=None, ctc_tokenizer=None, is_mustc=False
    ):
        super(SpeechTextDatatset).__init__()
        self.data_file = data_file
        self.target_sample_hz = target_sample_hz
        self.max_seconds = max_seconds
        self.max_length = self.target_sample_hz * max_seconds
        self.audios_files, self.text_files = [], []
        self.lengths = []
        self.return_all = return_all
        # TODO : implement a version that chunk audio into overlapping segments
        self.overlap_audio = overlap_audio
        self.lm_tokenizer = lm_tokenizer
        self.ctc_tokenizer = ctc_tokenizer
        self.normalize = normalize
        self.is_mustc = is_mustc

        with open(data_file, 'r') as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                if len(line) > 0:
                    try:
                        filepath, length, text_path = line.strip().split('\t')
                    except:
                        continue
                    self.audios_files.append(filepath)
                    self.text_files.append(text_path)
                    self.lengths.append(int(length))
        print(f"Loaded {len(self.audios_files)} audio files")


    @staticmethod
    def remove_special_characters(input_text):
        chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
        out = re.sub(chars_to_ignore_regex, '', input_text)
        return out

    @staticmethod
    def prepare_token(tokenizer, text):
        if tokenizer is None:
            return torch.zeros(1), 1
        else:
            tokenized_text = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
            tokenized_text = torch.cat([torch.tensor([tokenizer.bos_token_id]), tokenized_text,
                                        torch.tensor([tokenizer.eos_token_id])])
            text_length = tokenized_text.size(0)
            return tokenized_text, text_length

    def __getitem__(self, index):
        file = self.audios_files[index]
        text_file = self.text_files[index]
        data, sample_rate = torchaudio.load(file)
        if not self.is_mustc:
            with open(text_file, 'r') as f:
                text = f.read().strip()
        else:
            text = text_file # we directly put the data inside the path file
        if self.normalize:
            norm_text = self.remove_special_characters(text)
        else:
            norm_text = text

        lm_token, lm_token_len = self.prepare_token(self.lm_tokenizer, text)
        ctc_token, ctc_token_len = self.prepare_token(self.ctc_tokenizer, norm_text)
        if data.shape[0] > 1:
            # the audio has more than 1 channel, convert to mono
            data = reduce(data, 'c ... -> 1 ...', 'mean')

        data = resample(data, sample_rate, self.target_sample_hz)
        audio_length = data.size(1)
        # data is pre-processed to cap at the max-length so no need for checking anymore
        # assert audio_length <= self.max_length
        # remove the first dimension as it's always 1
        data = rearrange(data, '1 ... -> ...')
        lm_pad, ctc_pad = 0, 0
        if self.lm_tokenizer is not None:
            lm_pad = self.lm_tokenizer.pad_token_id
        if self.ctc_tokenizer is not None:
            ctc_pad = self.ctc_tokenizer.pad_token_id
        return data, audio_length, lm_token, lm_token_len, ctc_token, ctc_token_len, lm_pad, ctc_pad

    def __len__(self):
        return len(self.audios_files)

    def get_lengths(self):
        return self.lengths


class ASRDataModule(LightningDataModule):
    def __init__(self, args, tokenizer=None, lm_tokenizer=None, ctc_tokenizer=None):
        super().__init__()
        self.train_file = args.train_file
        self.valid_file = args.valid_file
        self.test_file = args.test_file
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.target_sample_hz = args.target_sample_hz
        self.max_seconds = args.max_seconds
        self.tokenizer = tokenizer
        # bpe level, no need for preprocessing
        self.normalize = False
        if args.tokenizer_vocab_path is None or args.use_char_ctc:
            self.normalize = True

        self.lm_tokenizer = lm_tokenizer
        self.ctc_tokenizer = ctc_tokenizer
        self.is_mustc = args.is_mustc


    def setup(self, stage: str) -> None:
        self.train_dataset = SpeechTextDatatset(
            self.train_file, self.target_sample_hz,
            max_seconds=self.max_seconds,
            lm_tokenizer=self.lm_tokenizer,
            ctc_tokenizer=self.ctc_tokenizer,
            normalize=self.normalize,
            is_mustc=self.is_mustc
        )
        self.valid_dataset = SpeechTextDatatset(
            self.valid_file, self.target_sample_hz,
            max_seconds=self.max_seconds,
            limit=1000,
            normalize=self.normalize,
            lm_tokenizer=self.lm_tokenizer,
            ctc_tokenizer=self.ctc_tokenizer,
            is_mustc=self.is_mustc
        )
        self.test_dataset = SpeechTextDatatset(
            self.test_file,
            self.target_sample_hz,
            max_seconds=self.max_seconds,
            limit=1000,
            normalize=self.normalize,
            return_all=True,
            lm_tokenizer=self.lm_tokenizer,
            ctc_tokenizer=self.ctc_tokenizer,
            is_mustc=self.is_mustc
        )

    @staticmethod
    def collate_fn(batch):
        """
        batch of: audio_data, audio_length, lm token, lm len, ctc token, ctc len, lm pad, ctc pad
        """
        batch_size = len(batch)
        audio_lengths = [_[1] for _ in batch]
        lm_lengths = [_[3] for _ in batch]
        ctc_lengths = [_[5] for _ in batch]
        lm_pad_token_id, ctc_pad_token_id = batch[0][6], batch[0][7]
        max_audio_len = max(audio_lengths)
        max_lm_len = max(lm_lengths)
        max_ctc_len = max(ctc_lengths)

        # audio is always zero-padded
        padded_audio_seq = torch.zeros((batch_size, max_audio_len))
        padded_lm_seq = torch.zeros((batch_size, max_lm_len)).fill_(lm_pad_token_id).long()
        padded_ctc_seq = torch.zeros((batch_size, max_ctc_len)).fill_(ctc_pad_token_id).long()
        audio_attention_amsk = torch.zeros((batch_size, max_audio_len))
        lm_attention_mask = torch.zeros((batch_size, max_lm_len)).long()
        ctc_attention_mask = torch.zeros((batch_size, max_ctc_len)).long()
        lm_labels = torch.zeros(batch_size, max_lm_len).fill_(IGNORE_INDEX).long()
        ctc_labels = torch.zeros(batch_size, max_ctc_len).fill_(IGNORE_INDEX).long()

        for i in range(batch_size):
            cur_audio, cur_audio_len, cur_lm_text, cur_lm_len, cur_ctc_text, cur_ctc_len, _, _ = batch[i]
            # use right pad
            padded_audio_seq[i, :cur_audio_len] = cur_audio
            audio_attention_amsk[i, : cur_audio_len] = 1
            padded_lm_seq[i, :cur_lm_len] = cur_lm_text
            lm_attention_mask[i, : cur_lm_len] = 1
            lm_labels[i, :cur_lm_len] = cur_lm_text
            padded_ctc_seq[i, :cur_ctc_len] = cur_ctc_text
            ctc_attention_mask[i, : cur_ctc_len] = 1
            ctc_labels[i, :cur_ctc_len] = cur_ctc_text
        return {
            "audio_seq": padded_audio_seq,
            "lm_text_seq": padded_lm_seq,
            "ctc_text_seq": padded_ctc_seq,
            "audio_attention_mask": audio_attention_amsk,
            "lm_attention_mask": lm_attention_mask,
            "ctc_attention_mask": ctc_attention_mask,
            "audio_length": audio_lengths,
            "lm_length": lm_lengths,
            "ctc_length": ctc_lengths,
            "lm_labels": lm_labels,
            "ctc_labels": ctc_labels
        }

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataloader = DataLoader(
            self.train_dataset,
            sampler=SpeechSampler(
                batch_size=self.batch_size, lengths=self.train_dataset.get_lengths()
            ),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )
        return dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dataloader = DataLoader(
            self.valid_dataset,
            sampler=SpeechSampler(
                batch_size=self.batch_size, lengths=self.valid_dataset.get_lengths()
            ),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )
        return dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return dataloader

