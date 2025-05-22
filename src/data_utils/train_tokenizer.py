# code to train BPE tokenizer for Speech Translation/Transcription
import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
from tokenizers import Tokenizer,pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


@dataclass
class ProcessArguments:
    data_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default=None)
    dataset_name : Optional[str] = field(default=None)
    split: Optional[str] = field(default=None)


def main(args):
    # aggregate datafiles
    train_files = []
    for split in ["train", "test", "valid"]:
        root_file = f"{args.data_dir}/{split}.txt"
        with open(root_file, "r") as f:
            for line in f:
                text_file_path = line.strip().split('\t')[-1]
                train_files.append(text_file_path)

    unk_token = "<unk>"  # token for unknown words
    spl_tokens = ["<unk>", "<pad>", "<s>", "</s>", "|"]  # special tokens
    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    trainer = BpeTrainer(special_tokens=spl_tokens, vocab_size=10000, show_progress=True)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train(trainer=trainer, files=train_files)
    os.makedirs(f"{args.output_dir}/{args.dataset_name}/tokenizer/", exist_ok=True)
    tokenizer.model.save(f"{args.output_dir}/{args.dataset_name}/tokenizer/")
    tokenizer.save(f"{args.output_dir}/{args.dataset_name}/tokenizer.json")

if __name__ == "__main__":
    args = HfArgumentParser(ProcessArguments).parse_args_into_dataclasses()[0]
    main(args)