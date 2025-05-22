from typing import Any
import pytorch_lightning as pl
import sacrebleu
import wandb
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datasets
from transformers import AdamW
from transformers.optimization import get_polynomial_decay_schedule_with_warmup
from src.models.w2v_simul import W2VSimulModel


class SimulModelModule(pl.LightningModule):
    def __init__(
            self, args,
            lm_tokenizer=None,
            ctc_tokenizer=None,
            lm_info=None,
            ctc_info=None
         ):
        super().__init__()
        self.args = args
        self.use_cif = args.use_cif
        self.model = W2VSimulModel(
            lm_info=lm_info,
            ctc_info=ctc_info,
            hidden_size=args.hidden_size,
            target_sample_hz=args.target_sample_hz,
            use_nuggets=args.use_nuggets,
            use_cif=self.use_cif,
            nugget_compress_rate=args.nugget_compress_rate,
            cache_dir=args.cache_dir,
            use_ilk=args.use_ilk,
            max_position=args.max_position
        )
        self.is_pretrain = args.is_pretrain
        self.lm_tokenizer = lm_tokenizer
        self.ctc_tokenizer = ctc_tokenizer
        self.batch_size = args.batch_size
        self.max_steps = args.max_steps
        self.target_sample_hz = args.target_sample_hz
        self.max_seconds = args.max_seconds
        self.learning_rate = args.learning_rate
        self.apply_grad_penalty_every = args.apply_grad_penalty_every
        self.output_dir = args.output_dir
        # for manual grad acc
        self.nugget_acc = 0
        self.grad_accum_every = args.grad_accum_every
        self.manual_val_loss, self.manual_nugget_loss = [], []
        self.val_lm_pred_text, self.val_lm_ref_text = [], []
        self.val_ctc_pred_text, self.val_ctc_ref_text = [], []
        self.use_nuggets = args.use_nuggets
        self.use_cif = args.use_cif
        self.nuggets_pretrain_steps = args.nuggets_pretrain_steps
        self.lm_wer_metric = datasets.load_metric("wer")
        self.ctc_wer_metric = datasets.load_metric("wer")
        self.is_training_nuggets = self.use_nuggets
        self.is_training_ctc = False
        if self.is_pretrain:
            print("Pre-training Causal LM based on Previous trained LM")
            print("First unfreeze w2v and train for 10k steps")
            print("Then freeze w2v and train ctc for 10k steps")
            print("This checkpoint would be the backbone for future nuggets and cif models")
            assert not self.use_nuggets
        if self.use_nuggets and not self.use_cif:
            print("Training Nuggets for Simul ST")
        if self.use_cif:
            print("Training CIF for Simul ST")


    @staticmethod
    def get_total_params_sum(model):
        total_params_sum = 0.0
        for param in model.parameters():
            total_params_sum += param.sum().item()
        return total_params_sum

    @staticmethod
    def get_total_grad_norm(model):
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.requires_grad:
                total_grad_norm += param.grad.norm().item() ** 2
        return total_grad_norm


    def pretrain_step(self, batch, batch_idx, **kwargs):
        w2v_optimizer, ctc_optimizer, encoder_optimizer, decoder_optimizer = self.optimizers()
        w2v_scheduler, ctc_scheduler, encoder_scheduler, decoder_scheduler = self.lr_schedulers()
        pretrian_steps = 2000000 # nvr train ctc lol, as we find it unuseful
        if self.trainer.global_step == pretrian_steps:
            print("Pretrain Causal W2V finished, start CTC training")
            self.is_training_ctc = True
        audio_seq = batch["audio_seq"]
        audio_attn_mask = batch["audio_attention_mask"]  # mask for audio tokens
        lm_text_seq = batch["lm_text_seq"]
        lm_attn_mask = batch["lm_attention_mask"]
        lm_labels = batch["lm_labels"]
        ctc_text_seq = batch["ctc_text_seq"]
        ctc_attn_mask = batch["ctc_attention_mask"]
        ctc_labels = batch["ctc_labels"]
        # loss is the avg of all loses, misc_info contains breakdown of loss and other info
        loss, misc_info = self.model(
            audio_seq, audio_attn_mask,
            lm_text_seq, lm_attn_mask, lm_labels,
            ctc_text_seq, ctc_attn_mask, ctc_labels,
            is_training_nuggets=False,
            is_training_ctc=self.is_training_ctc,
        )
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for key in misc_info.keys():
            # log loss breakdown for better visualization
            if "loss" in key and misc_info[key] is not None:
                self.log(key, misc_info[key], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.nugget_acc += 1
        self.manual_backward(loss)
        self.clip_gradients(w2v_optimizer, 1, "norm")
        self.clip_gradients(encoder_optimizer, 1, "norm")
        # make text decoder relatively more stable
        self.clip_gradients(decoder_optimizer, 0.5, "norm")
        if self.nugget_acc % self.grad_accum_every == 0:
            self.nugget_acc = 0
            if self.trainer.global_step > pretrian_steps:
                # causal w2v is trained, now train ctc
                ctc_optimizer.step()
                ctc_optimizer.zero_grad()
            else:
                encoder_optimizer.step()
                decoder_optimizer.step()
                encoder_scheduler.step()
                decoder_scheduler.step()
                w2v_optimizer.step()
                w2v_optimizer.zero_grad()
                w2v_scheduler.step()

        return {"loss": loss}

    def training_step(self, batch, batch_idx, **kwargs):
        if self.trainer.global_step == 0:
            print(batch)
        if self.use_nuggets and self.trainer.global_step == self.nuggets_pretrain_steps:
            self.is_training_nuggets = False
            print("Finished nugget pretraining, now training the whole model")

        if self.is_pretrain:
            # no nugget or other compression is involved
            return self.pretrain_step(batch, batch_idx, **kwargs)

        if self.use_nuggets:
            w2v_optimizer, encoder_optimizer, decoder_optimizer, nuggets_optimizer = self.optimizers()
            w2v_scheduler, encoder_scheduler, decoder_scheduler, nuggets_scheduler = self.lr_schedulers()
        else:
            w2v_optimizer, encoder_optimizer, decoder_optimizer = self.optimizers()
            w2v_scheduler, encoder_scheduler, decoder_scheduler = self.lr_schedulers()

        audio_seq = batch["audio_seq"]
        audio_attn_mask = batch["audio_attention_mask"]  # mask for audio tokens
        lm_text_seq = batch["lm_text_seq"]
        lm_attn_mask = batch["lm_attention_mask"]
        lm_labels = batch["lm_labels"]
        ctc_text_seq = batch["ctc_text_seq"]
        ctc_attn_mask = batch["ctc_attention_mask"]
        ctc_labels = batch["ctc_labels"]
        # loss is the avg of all loses, misc_info contains breakdown of loss and other info
        loss, misc_info = self.model(
            audio_seq, audio_attn_mask,
            lm_text_seq, lm_attn_mask, lm_labels,
            ctc_text_seq, ctc_attn_mask, ctc_labels,
            is_training_nuggets=self.is_training_nuggets,
        )
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for key in misc_info.keys():
            # log loss breakdown for better visualization
            if "loss" in key and misc_info[key] is not None:
                self.log(key, misc_info[key], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if misc_info["ratio"] is not None:
            self.log("nugget_ratio", misc_info["ratio"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.nugget_acc += 1
        self.manual_backward(loss)
        if self.use_nuggets:
            self.clip_gradients(nuggets_optimizer, 1, "norm")
        self.clip_gradients(encoder_optimizer, 1, "norm")
        # make text decoder relatively more stable
        self.clip_gradients(decoder_optimizer, 0.5, "norm")
        self.clip_gradients(w2v_optimizer, 1, "norm")
        if self.nugget_acc % self.grad_accum_every == 0:
            self.nugget_acc = 0
            if self.use_nuggets and self.is_training_nuggets:
                nuggets_optimizer.step()
                nuggets_optimizer.zero_grad()
            w2v_optimizer.step()
            w2v_optimizer.zero_grad()
            w2v_scheduler.step()

            encoder_optimizer.step()
            decoder_optimizer.step()
            encoder_scheduler.step()
            decoder_scheduler.step()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

        return {"loss": loss}

    def validation_step(self, batch, batch_idx, **kwargs):
        audio_seq = batch["audio_seq"]
        audio_attn_mask = batch["audio_attention_mask"]  # mask for audio tokens
        lm_text_seq = batch["lm_text_seq"]
        lm_attn_mask = batch["lm_attention_mask"]
        lm_labels = batch["lm_labels"]
        ctc_text_seq = batch["ctc_text_seq"]
        ctc_attn_mask = batch["ctc_attention_mask"]
        ctc_labels = batch["ctc_labels"]
        val_loss, misc_info = self.model(
            audio_seq, audio_attn_mask,
            lm_text_seq, lm_attn_mask, lm_labels,
            ctc_text_seq, ctc_attn_mask, ctc_labels,
            is_training_nuggets=self.is_training_nuggets,
            is_training_ctc=self.is_training_ctc
        )

        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        for key in misc_info.keys():
            # log loss breakdown for better visualization
            if "loss" in key and misc_info[key] is not None:
                val_key = f"val_{key}"
                self.log(val_key, misc_info[key], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if misc_info["ratio"] is not None:
            self.log("val_nugget_ratio", misc_info["ratio"], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # bz x gen_len, bz x nugget_len
        lm_gen, ctc_gen, misc_info = self.model(
            audio_seq, audio_attn_mask,
            lm_text_seq, lm_attn_mask, lm_labels,
            ctc_text_seq, ctc_attn_mask, ctc_labels,
            inference=True,
            is_training_nuggets=self.is_training_nuggets,
            is_training_ctc=self.is_training_ctc
        )

        lm_pred_text, lm_ref_text = self.post_process_generation(lm_gen, self.lm_tokenizer, lm_text_seq, group_tokens=False)
        ctc_pred_text, ctc_ref_text = self.post_process_generation(ctc_gen, self.ctc_tokenizer, ctc_text_seq, group_tokens=True)

        if lm_gen is not None:
            self.val_lm_pred_text.extend(lm_pred_text)
            self.val_lm_ref_text.extend(lm_ref_text)
            self.lm_wer_metric.add_batch(predictions=lm_pred_text, references=lm_ref_text)
        if ctc_gen is not None:
            self.val_ctc_pred_text.extend(ctc_pred_text)
            self.val_ctc_ref_text.extend(ctc_ref_text)
            self.ctc_wer_metric.add_batch(predictions=ctc_pred_text, references=ctc_ref_text)

        if batch_idx == 0 and self.global_rank == 0 and isinstance(self.logger, WandbLogger):
            # print(misc_info["nugget_indices"])
            self.log_wandb_info(
                lm_pred_text, lm_ref_text,
                ctc_pred_text, ctc_ref_text,
                audio_seq, misc_info["nugget_indices"]
            )

        return {"val_loss", val_loss}

    def log_wandb_info(self, lm_pred_text, lm_ref_text, ctc_pred_text, ctc_ref_text, audio_seq, nugget_indices):
        bz = audio_seq.shape[0]
        log_data = []
        columns = ["lm_pred_text", "lm_ref_text", "ctc_pred_text", "ctc_ref_text"]
        if nugget_indices is not None:
            columns.extend(["nugget_indices"])

        for i in range(bz):
            cur_lm_pred_text = lm_pred_text[i] if lm_pred_text else ""
            cur_lm_ref_text = lm_ref_text[i] if lm_ref_text else ""
            cur_ctc_pred_text = ctc_pred_text[i] if ctc_pred_text else ""
            cur_ctc_ref_text = ctc_ref_text[i] if ctc_ref_text else ""
            cur_data = [cur_lm_pred_text, cur_lm_ref_text, cur_ctc_pred_text, cur_ctc_ref_text]
            if nugget_indices is not None:
                cur_nugget_indices = nugget_indices[i, :].cpu().detach().numpy() * 320
                cur_step = self.trainer.global_step
                self.plot_segmentation(cur_nugget_indices, audio_seq[i, :].cpu().detach().numpy(),
                                       f"{self.output_dir}/sample_{cur_step}_{i}_nuggets.png")
                nugget_image = wandb.Image(f"{self.output_dir}/sample_{cur_step}_{i}_nuggets.png", caption="nugget segmentation")
                cur_data.append(nugget_image)
            log_data.append(cur_data)
        self.logger.log_table(key="asr_table", columns=columns, data=log_data)

    @staticmethod
    def post_process_generation(gen_tokens, tokenizer, labels, group_tokens=False):
        pred_text, ref_text = None, None
        if gen_tokens is not None:
            pred_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True, group_tokens=group_tokens)
            ref_text = tokenizer.batch_decode(labels, skip_special_tokens=True, group_tokens=group_tokens)
        return pred_text, ref_text


    def plot_segmentation(self, nugget_indices, orig_audio, filename):
        # nugget indice has shape (nugget_len, )
        # orig audio has shape (seq_len, )
        orig_audio = orig_audio.squeeze()
        x_values = list(range(orig_audio.shape[0]))
        plt.plot(x_values, orig_audio.squeeze(), label="speech sequence")
        for index in nugget_indices.tolist():
            # padding index is -1
            if index > 0:
                plt.axvline(x=index, color='r', linestyle='--')

        # Label your plot
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Audio Sequence with Highlighted Indices')
        # plt.show()
        plt.savefig(filename)
        plt.clf()

    def on_validation_epoch_end(self) -> None:
        if len(self.val_lm_ref_text) > 0:
            lm_wer = self.lm_wer_metric.compute()
            lm_bleu = sacrebleu.corpus_bleu(self.val_lm_pred_text, [self.val_lm_ref_text]).score
            self.log("lm_wer", lm_wer, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("bleu", lm_bleu, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.val_lm_pred_text, self.val_lm_ref_text = [], []
        if len(self.val_ctc_ref_text) > 0:
            ctc_wer = self.ctc_wer_metric.compute()
            self.log("ctc_wer", ctc_wer, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.val_ctc_pred_text, self.val_ctc_ref_text = [], []

    def test_step(self, batch, batch_idx, **kwargs):
        audio_seq = batch["audio_seq"]
        audio_attn_mask = batch["audio_attention_mask"]  # mask for audio tokens
        lm_text_seq = batch["lm_text_seq"]
        lm_attn_mask = batch["lm_attention_mask"]
        lm_labels = batch["lm_labels"]
        ctc_text_seq = batch["ctc_text_seq"]
        ctc_attn_mask = batch["ctc_attention_mask"]
        ctc_labels = batch["ctc_labels"]
        if audio_seq.shape[1] > 16000 * 20:
            lm_gen = None
        else:
            lm_gen, ctc_gen, misc_info = self.model(
                audio_seq, audio_attn_mask,
                lm_text_seq, lm_attn_mask, lm_labels,
                ctc_text_seq, ctc_attn_mask, ctc_labels,
                inference=True,
                is_training_nuggets=False,
                is_training_ctc=False
            )
        lm_pred_text, lm_ref_text = self.post_process_generation(lm_gen, self.lm_tokenizer, lm_text_seq,  group_tokens=False)
        print(lm_pred_text)
        print(lm_ref_text)
        # we only look at lm for inference
        # ctc_pred_text, ctc_ref_text = self.post_process_generation(ctc_gen, self.ctc_tokenizer, ctc_text_seq, group_tokens=True)
        if lm_gen is not None:
            self.val_lm_pred_text.extend(lm_pred_text)
            self.val_lm_ref_text.extend(lm_ref_text)
            self.lm_wer_metric.add_batch(predictions=lm_pred_text, references=lm_ref_text)

    def on_test_epoch_end(self) -> None:
        wer = self.lm_wer_metric.compute()
        bleu_score = sacrebleu.corpus_bleu(self.val_lm_pred_text, [self.val_lm_ref_text]).score
        print("BLEU Score: ", bleu_score)
        print("WER Score: ", wer)




    def configure_pretrain_optimizers(self):
        warmup_steps = 6000
        w2v_params, ctc_params = [], []
        encoder_params, decoder_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            elif "semantic_encoder" in name:
                encoder_params.append(param)
            elif "block_attn" in name:
                # although this is only used when nugget is activated, we treat it as part of semantic repr
                encoder_params.append(param)
            elif "encoder" in name:
                # w2v is not frozen, need to train a decoder only version
                # print("add", name)
                w2v_params.append(param)
            elif "ctc_head" in name:
                ctc_params.append(param)
            elif "lm_decoder" in name:
                decoder_params.append(param)

        w2v_optimizer = AdamW(w2v_params, lr=self.learning_rate)
        ctc_optimizer = AdamW(ctc_params, lr=self.learning_rate)
        encoder_optimizer = AdamW(encoder_params, lr=self.learning_rate)
        decoder_optimizer = AdamW(decoder_params, lr=self.learning_rate)
        w2v_optimizer_config = {
            "optimizer": w2v_optimizer,
            "lr_scheduler": {
                "scheduler": get_polynomial_decay_schedule_with_warmup(
                    w2v_optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=self.max_steps,
                    lr_end=3e-5,
                ),
            },
        }

        encoder_optimizer_config = {
            "optimizer": encoder_optimizer,
            "lr_scheduler": {
                "scheduler": get_polynomial_decay_schedule_with_warmup(
                    w2v_optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=self.max_steps,
                    lr_end=3e-5,
                ),
            },
        }

        decoder_optimizer_config = {
            "optimizer": decoder_optimizer,
            "lr_scheduler": {
                "scheduler": get_polynomial_decay_schedule_with_warmup(
                    w2v_optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=self.max_steps,
                    lr_end=3e-5,
                ),
            },
        }



        ctc_optimizer_config = {
            "optimizer": ctc_optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    ctc_optimizer,
                    mode="min",
                    factor=0.5,
                    patience=8,
                    min_lr=3e-5,
                    verbose=True,
                ),
            },
        }
        return w2v_optimizer_config, ctc_optimizer_config, encoder_optimizer_config, decoder_optimizer_config


    def configure_optimizers(self):
        if self.is_pretrain:
            return self.configure_pretrain_optimizers()
        warmup_steps=6000
        nuggets_param, w2v_params, encoder_params, decoder_params = [], [], [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            elif "nugget" in name:
                nuggets_param.append(param)
            elif "semantic_encoder" in name:
                # encoder is the semantic encoder we added
                encoder_params.append(param)
            elif "encoder" in name:
                # w2v is not frozen, need to train a decoder only version
                # print("add", name)
                w2v_params.append(param)
            elif "lm_decoder" in name:
                decoder_params.append(param)

        encoder_optimizer = AdamW(encoder_params, lr=self.learning_rate)
        decoder_optimizer = AdamW(decoder_params, lr=self.learning_rate)
        w2v_optimizer = AdamW(w2v_params, lr=self.learning_rate)

        encoder_optimizer_config = {
            "optimizer": encoder_optimizer,
            "lr_scheduler": {
                "scheduler": get_polynomial_decay_schedule_with_warmup(
                    encoder_optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=self.max_steps,
                    lr_end=3e-5,
                ),
            },
        }

        decoder_optimizer_config = {
            "optimizer": decoder_optimizer,
            "lr_scheduler": {
                "scheduler": get_polynomial_decay_schedule_with_warmup(
                    decoder_optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=self.max_steps,
                    lr_end=3e-5,
                ),
            },
        }

        w2v_optimizer_config = {
            "optimizer": w2v_optimizer,
            "lr_scheduler": {
                "scheduler": get_polynomial_decay_schedule_with_warmup(
                    w2v_optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=self.max_steps,
                    lr_end=3e-5,
                ),
            },
        }

        if not self.use_nuggets:
            return w2v_optimizer_config, encoder_optimizer_config, decoder_optimizer_config

        nuggets_optimizer = AdamW(nuggets_param, lr=1e-4)
        nuggets_optimizer_config = {
            "optimizer": nuggets_optimizer,
            "lr_scheduler": {
                 "scheduler": ReduceLROnPlateau(
                    nuggets_optimizer,
                    mode="min",
                    factor=0.5,
                    patience=8,
                    min_lr=3e-5,
                    verbose=True,
                ),
            },
        }

        return w2v_optimizer_config, encoder_optimizer_config, decoder_optimizer_config, nuggets_optimizer_config

    @property
    def automatic_optimization(self) -> bool:
        return False

