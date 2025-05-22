import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional
import torch
import pytorch_lightning as pl
import transformers
from pytorch_lightning import Trainer as pl_Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from tokenizers import Tokenizer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from transformers import HfArgumentParser
from src.models.my_wav2vec import MyWav2Vec2CTCTokenizer, MyLMTokenizer
from src.data_utils.data_module import ASRDataModule
from src.lightning.wav2vec_trainer import W2VModelModule

logging.basicConfig(level=logging.DEBUG)


@dataclass
class ModelArguments:
    cache_dir: Optional[str] = field(default="./cache")
    model_path: Optional[str] = field(default=None)
    ctc_pretrained_path: Optional[str] = field(default=None)
    mode: Optional[str] = field(default="train")
    tokenizer_vocab_path: Optional[str] = field(default=None)
    local_rank: Optional[int] = field(default=0)
    train_file: Optional[str] = field(default=None)  # {filepath length}
    valid_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default=None)
    target_sample_hz: Optional[int] = field(default=16000)
    streaming: Optional[bool] = field(default=False)
    shuffle_buffer: Optional[int] = field(default=5000)
    output_dir: Optional[str] = field(default="./checkpoints")

    hidden_size: Optional[int] = field(default=768)
    max_steps: Optional[int] = field(default=100000)
    batch_size: Optional[int] = field(default=2)
    grad_accum_every: Optional[int] = field(default=1)
    max_seconds: Optional[int] = field(default=3)
    strategy: Optional[str] = field(default="ddp")
    use_ctc: Optional[bool] = field(default=False)
    use_lm: Optional[bool] = field(default=False)
    use_char_ctc: Optional[bool] = field(default=False)

    learning_rate: Optional[float] = field(default=2e-4)
    lr_scheduler_type: Optional[str] = field(default="cosine")
    num_warmup_steps: Optional[int] = field(default=100)
    weight_decay: Optional[float] = field(default=0.05)
    apply_grad_penalty_every: Optional[int] = field(default=4)

    num_gpus: Optional[int] = field(default=1)
    log_freq: Optional[int] = field(default=10)
    eval_freq: Optional[int] = field(default=20)
    save_freq: Optional[int] = field(default=30)

    use_nuggets: Optional[bool] = field(default=False)
    use_cif: Optional[bool] = field(default=False)
    nuggets_pretrain_steps: Optional[int] = field(default=100000)
    nugget_window_size: Optional[int] = field(default=64)
    disable_nuggets: Optional[bool] = field(default=False)
    nugget_compress_rate: Optional[int] = field(default=12)
    dynamic_compress_rate: Optional[bool] = field(default=False)
    use_ctc_as_scorer: Optional[bool] = field(default=False)
    ctc_pretrain_steps : Optional[int] = field(default=5000)
    kl_loss_scale : Optional[float] = field(default=100.0)
    entropy_loss_scale : Optional[float] = field(default=0.1)

    run_name: Optional[str] = field(default=None)
    seed: Optional[int] = field(default=42)
    num_workers: Optional[int] = field(default=10)
    is_mustc: Optional[bool] = field(default=False)
    tune_w2v: Optional[bool] = field(default=False)


class W2VTrainer:
    @staticmethod
    def prepare_tokenizer(args):
        lm_tokenizer, ctc_tokenizer = None, None
        if args.use_ctc:
            # init from pre-trained word level ctc, data needs to be normalized as well
            ctc_tokenizer = MyWav2Vec2CTCTokenizer(vocab_file=args.tokenizer_vocab_path, unk_token="<unk>",
                                                    pad_token="<pad>", word_delimiter_token="|")
        if args.use_lm:
            assert args.tokenizer_vocab_path is not None, "Please provide a tokenizer vocab file for LM training"

            lm_tokenizer = MyLMTokenizer(
                Tokenizer.from_file(args.tokenizer_vocab_path.replace("vocab.json", "tokenizer.json")))
        return lm_tokenizer, ctc_tokenizer

    @staticmethod
    def get_tokenizer_info(tokenizer, is_ctc=False):
        if tokenizer is None:
            return None
        ret = dict()
        ret["bos_token_id"] = tokenizer.bos_token_id
        ret["eos_token_id"] = tokenizer.eos_token_id
        ret["unk_token_id"] = tokenizer.unk_token_id
        ret["pad_token_id"] = tokenizer.pad_token_id
        ret["vocab_size"] = tokenizer.vocab_size
        if is_ctc:
            ret["word_delimiter_token_id"] = tokenizer.word_delimiter_token_id
        return ret

    def __init__(self, args, mode="train") -> None:
        self.args = args
        self.print_important_params(args)
        self.lm_tokenizer, self.ctc_tokenizer = self.prepare_tokenizer(args)
        ctc_info = self.get_tokenizer_info(self.ctc_tokenizer, is_ctc=True)
        lm_info = self.get_tokenizer_info(self.lm_tokenizer, is_ctc=False)
        self.lightning_data_module = ASRDataModule(
            args,
            lm_tokenizer=self.lm_tokenizer,
            ctc_tokenizer=self.ctc_tokenizer
        )
        accelerator = "cpu" if int(args.num_gpus) == 0 else "gpu"
        if mode == "eval":
            assert args.model_path is not None, "Please provide a model path for evaluation"
            print(f"Loading ckpt: {args.model_path}")
            self.lightning_model_module = W2VModelModule.load_from_checkpoint(
                checkpoint_path=args.model_path,
                args=args,
                strict=True,
                lm_tokenizer=self.lm_tokenizer,
                ctc_tokenizer=self.ctc_tokenizer,
                lm_info=lm_info,
                ctc_info=ctc_info
            )
            print("Finish Loading Soundstream Model!")
            print(self.lightning_model_module.model)
            self.trainer = pl_Trainer(
                enable_progress_bar=True,
                devices=int(args.num_gpus) if int(args.num_gpus) > 0 else None,
                accelerator=accelerator,
            )

        else:
            model_checkpoint = ModelCheckpoint(
                dirpath=args.output_dir,
                monitor="val_loss",
                # filename="{epoch:02d}-{val_loss:.2f}",
                every_n_train_steps=args.save_freq,
                save_top_k=5,
                verbose=True,
                save_on_train_epoch_end=True,
            )
            if args.model_path is not None:
                print("Loading ckpt: ", args.model_path)
                self.lightning_model_module = W2VModelModule.load_from_checkpoint(
                    checkpoint_path=args.model_path,
                    args=args,
                    strict=False,
                    lm_tokenizer=self.lm_tokenizer,
                    ctc_tokenizer=self.ctc_tokenizer,
                    lm_info=lm_info,
                    ctc_info=ctc_info
                )
                if args.ctc_pretrained_path is not None:
                    print("Loading ctc ckpt: ", args.ctc_pretrained_path)
                    ctc_ckpt_weight = torch.load(args.ctc_pretrained_path)["state_dict"]
                    ctc_weight_to_load = dict()
                    for key, value in ctc_ckpt_weight.items():
                        if "ctc_head" in key:
                            ctc_weight_to_load[key[len("model.ctc_head."):]] = value
                    self.lightning_model_module.model.ctc_head.load_state_dict(
                        ctc_weight_to_load,
                        strict=True
                    )
            else:
                self.lightning_model_module = W2VModelModule(
                    args,
                    lm_tokenizer=self.lm_tokenizer,
                    ctc_tokenizer=self.ctc_tokenizer,
                    lm_info=lm_info,
                    ctc_info=ctc_info
                )
            lr_monitor = LearningRateMonitor(logging_interval="step")

            if args.run_name.startswith("debug"):
                wandb_logger = None
            else:
                random_string = str(int(time.time()))
                wandb_logger = WandbLogger(
                    project="SpeechNuggets",
                    # entity="hopkins-piggy-farm",
                    name=f"{args.run_name}-{random_string}",
                    save_dir=f"{args.output_dir}",
                )
            # wandb_logger.watch(self.lightning_model_module, log="all")
            if args.strategy == "ddp":
                strategy = DDPStrategy(find_unused_parameters=True, process_group_backend='nccl')

            self.trainer = pl_Trainer(
                max_steps=args.max_steps,
                log_every_n_steps=args.log_freq,
                val_check_interval=args.eval_freq * args.grad_accum_every,
                enable_progress_bar=True,
                # gradient_clip_val=1.0,
                devices=int(args.num_gpus) if int(args.num_gpus) > 0 else None,
                # accumulate_grad_batches=args.grad_accum_every,
                logger=wandb_logger,
                callbacks=[model_checkpoint, lr_monitor],
                accelerator=accelerator,
                enable_model_summary=True,
                strategy=strategy,
                # find_unused_parameters=True,
                # num_sanity_val_steps=-1,
                # detect_anomaly=True,
            )

    @staticmethod
    def print_important_params(args) -> None:
        print("#############################################")
        print(f"Use CTC: {args.use_ctc}\nUse Char CTC: {args.use_char_ctc}")
        print(f"Use LM: {args.use_lm}")
        print(f"Use Nuggets for Compression: {args.use_nuggets}, Compression Rate: {args.nugget_compress_rate}")
        print(f"Use CIF for Compression: {args.use_cif}")
        print(f"Nuggets pre-training steps: {args.nuggets_pretrain_steps}")
        print(f"Disable Nuggets: {args.disable_nuggets}")
        print(f"Use CTC as Scorer: {args.use_ctc_as_scorer}")
        print("Using Dynamic Compression Rate: ", args.dynamic_compress_rate)

        # some sanity check before training kicks off
        if args.use_ctc_as_scorer and args.disable_nuggets:
            if args.disable_nuggets:
                raise RuntimeError("Cannot use CTC as scorer and disable nuggets at the same time")
            if not args.use_char_ctc or (args.tokenizer_vocab_path is not None):
                raise RuntimeError("CTC scorer must be char level and use pre-trained ckpt from facebook/wav2vec2-base")

        print("#############################################")

    def train(self):
        self.trainer.fit(self.lightning_model_module, self.lightning_data_module)

    def test(self):
        self.trainer.test(model=self.lightning_model_module, datamodule=self.lightning_data_module)


if __name__ == "__main__":
    args = HfArgumentParser(ModelArguments).parse_args_into_dataclasses()[0]
    pl.seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    transformers.logging.set_verbosity_error()
    logging.basicConfig(level=logging.ERROR)
    print(args)
    trainer = W2VTrainer(args, mode=args.mode)
    if args.mode == "train":
        trainer.train()
    else:
        trainer.test()
