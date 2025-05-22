
from functools import partial
import torch
import torch.nn.functional as F
from einops import rearrange, pack
from torch import nn
from torchaudio.functional import resample
from transformers import GPT2Config, GenerationConfig
from src.models.utils import CausalConv1d
from src.models.my_wav2vec import MyWav2Vec2Model, MyWav2Vec2ForCTC
from src.models.my_transformer import MyGPT2ForLM, MyGPT2Model
from src.models.utils import ResidualUnit

def exists(val):
    return val is not None

def round_down_nearest_multiple(num, divisor):
    return num // divisor * divisor


def curtail_to_multiple(t, mult, from_left=False):
    data_len = t.shape[-1]
    rounded_seq_len = round_down_nearest_multiple(data_len, mult)
    seq_slice = slice(None, rounded_seq_len) if not from_left else slice(-rounded_seq_len, None)
    return t[..., seq_slice]


class W2VASRModel(nn.Module):
    def __init__(
            self,
            lm_info=None,
            ctc_info = None,
            hidden_size=512,
            target_sample_hz=16000,
            use_nuggets=False,
            nugget_compress_rate=12,
            dynamic_compress_rate=False,
            nugget_window_size=64,
            disable_nuggets=False,
            cache_dir=None,
            use_cif=False,
            use_ctc=False,
            use_ctc_as_scorer=False,
            use_lm=False,
            kl_loss_scale=100,
            entropy_loss_scale=0.1,
    ):
        super().__init__()
        self.use_cif = use_cif
        self.use_ctc = use_ctc
        self.use_ctc_as_scorer = use_ctc_as_scorer
        self.disable_nuggets = disable_nuggets
        self.dynamic_compress_rate = dynamic_compress_rate
        self.kl_loss_scale = kl_loss_scale
        self.entropy_loss_scale = entropy_loss_scale
        if self.dynamic_compress_rate:
            assert self.use_ctc_as_scorer, "dynamic_compress_rate must be used with CTC as scorer to infer segmentation"
        if self.use_ctc_as_scorer:
            assert self.use_ctc, "use_ctc_as_scorer must be used with use_ctc"
            assert not self.disable_nuggets, "use_ctc_as_scorer cannot be used with disable_nuggets (conv based nuggets)"
        self.use_lm = use_lm
        if (not use_ctc) and (not use_lm):
            raise RuntimeError("At least one of use_ctc and use_lm must be True")
        self.ctc_info = ctc_info
        self.lm_info = lm_info

        self.target_sample_hz = target_sample_hz  # for resampling on the fly
        self.nugget_window_size = nugget_window_size
        self.use_nuggets = use_nuggets
        self.disable_nuggets = disable_nuggets
        self.nugget_model = None
        self.hidden_size = hidden_size
        self.word2subword_ratio = 1.5

        self.encoder = MyWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=cache_dir)
        # encoder may-be partially frozen , we control it in optimizer
        self.encoder.freeze_feature_encoder()
        self.encoder.freeze_feature_extractor()
        # on top of frozen pre-trained wav2vec2 encoder, we add a decoder to extract semantic feature
        semantic_encoder_config = GPT2Config(
            n_embd=self.hidden_size,
            # the special token id and vocab size here does not matter
            # since semantic encoder is only dealing with audio feature
            vocab_size=4,
            num_labels=4,
            n_positions=1024,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            n_layer=4,
            n_head=8,
            add_cross_attention=False
        )
        self.semantic_encoder = MyGPT2Model(semantic_encoder_config)

        if self.use_nuggets:
            self.nugget_compress_rate = nugget_compress_rate
            self.strides = self.choose_strides_by_compress_rate(nugget_compress_rate)
            if disable_nuggets:
                assert not self.use_ctc, "When using conv compression, CTC loss cannot be computed"
                down_samplers = []
                residual_unit = partial(ResidualUnit, squeeze_excite=False, pad_mode='reflect')
                for stride in self.strides:
                    down_samplers.append(
                        residual_unit(self.hidden_size, self.hidden_size, 1, kernel_size=5)
                    )
                    down_samplers.append(
                        CausalConv1d(hidden_size, hidden_size, 2 * stride + 1, stride=stride),
                    )
                self.nugget_model = nn.Sequential(*down_samplers)
            else:
                # basically just to learn a scorer
                self.nugget_model = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_size, 1)
                )
        else:
            print("No compression mechanism is used")

        if self.use_ctc:
            assert ctc_info is not None, "Something wrong with ctc tokenizer"
            ctc_head_config = self.encoder.config
            ctc_head_config.vocab_size = ctc_info["vocab_size"]
            ctc_head_config.num_labels = ctc_info["vocab_size"]
            ctc_head_config.bos_token_id = ctc_info["bos_token_id"]
            ctc_head_config.eos_token_id = ctc_info["eos_token_id"]
            ctc_head_config.pad_token_id = ctc_info["pad_token_id"]
            self.ctc_head = MyWav2Vec2ForCTC(
                ctc_head_config,
            )

        if self.use_lm:
            lm_decoder_config = GPT2Config(
                n_embd=self.hidden_size,
                vocab_size=lm_info["vocab_size"],
                num_labels=lm_info["vocab_size"],
                n_positions=512,
                pad_token_id=lm_info["pad_token_id"],
                bos_token_id=lm_info["bos_token_id"],
                eos_token_id=lm_info["eos_token_id"],
                n_layer=4,
                n_head=8,
                add_cross_attention=True
            )
            self.lm_decoder = MyGPT2ForLM(lm_decoder_config)

    @staticmethod
    def choose_strides_by_compress_rate(compress_rate):
        if compress_rate < 8:
            return [compress_rate]
        elif compress_rate == 12:
            return [4, 3]
        elif compress_rate == 18:
            return [6, 3]
        elif compress_rate == 30:
            return [5, 3, 2]
        elif compress_rate == 50:
            return [5, 5, 2]
        elif compress_rate == 60:
            return [5, 4, 3]
        elif compress_rate == 120:
            return [6, 5, 4]

    @property
    def seq_len_multiple_of(self):
        if self.use_nuggets:
            return self.nugget_compress_rate * self.stride
        return self.stride

    @property
    def downsample_factor(self):
        return self.seq_len_multiple_of

    def process_input(
            self,
            x,
            input_sample_hz=None,
            curtail_from_left=False
    ):
        x, ps = pack([x], '* n')
        if exists(input_sample_hz):
            x = resample(x, input_sample_hz, self.target_sample_hz)

        x = curtail_to_multiple(x, self.seq_len_multiple_of, from_left=curtail_from_left)

        if x.ndim == 2:
            x = rearrange(x, 'b n -> b 1 n')

        return x, ps

    def apply_conv_compression(self, semantic_repr, attn_mask):
        compress_semantic_repr = self.nugget_model(semantic_repr.transpose(1, 2)).transpose(1, 2).contiguous()
        output_lengths = attn_mask.cumsum(dim=-1)[:, -1]
        for stride in self.strides:
            output_lengths = torch.div(output_lengths - 2 * stride - 1, stride, rounding_mode="floor") + 1
        output_lengths = output_lengths.long()
        compress_attn_mask = torch.zeros(
            (compress_semantic_repr.shape[0], compress_semantic_repr.shape[1]), dtype=attn_mask.dtype,
            device=attn_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        compress_attn_mask[
            (torch.arange(compress_attn_mask.shape[0], device=compress_attn_mask.device), output_lengths - 1)] = 1
        compress_attn_mask = compress_attn_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return compress_semantic_repr, compress_attn_mask

    def apply_nugget_compression(
            self, encoder_repr, attn_mask,
            nugget_scores=None, n_nugget=None
    ):
        """
        encoder_repr: (bz, seq_len, hid_dim) from acoustic encoder
        attn_mask: (bz, seq_len) from both encoder (as they have equal seq len)
        """
        # Step 1: compute nugget scores on acoustic repr (it's frozen repr)
        n_tok = attn_mask.sum(dim=1)
        if n_nugget is None:
            ratio = 1 / self.nugget_compress_rate
            n_nugget = (n_tok * ratio + 0.99).to(torch.int64)
            n_nugget[n_nugget > n_tok] = n_tok[n_nugget > n_tok]

        if nugget_scores is None:
            nugget_scores = self.nugget_model(encoder_repr).squeeze(-1)

        # Step 2: select nuggets based on scores (and force selection the last index)
        max_nugget = n_nugget.max()
        scores4nugget = nugget_scores.clone().detach()
        # shape (bsz, tok)
        scores4nugget[~attn_mask] = torch.finfo(scores4nugget.dtype).min
        n_tok_exp = (n_tok - 1).unsqueeze(1)
        scores4nugget = scores4nugget.scatter(
            1, n_tok_exp,
            scores4nugget.new_full(n_tok_exp.shape, fill_value=torch.finfo(scores4nugget.dtype).max)
        )
        # shape (bsz, nugget)
        nugget_mask = torch.arange(max_nugget, device=scores4nugget.device).unsqueeze(0) < n_nugget.unsqueeze(1)
        ind = scores4nugget.argsort(1, descending=True)[:, :max_nugget]
        # sort the indices
        ind4sort = ind * nugget_mask + (ind + attn_mask.shape[1] + 1) * (~nugget_mask)
        resort_ind = ind4sort.argsort(1)
        indices_ascending = ind.gather(1, resort_ind)
        scores_gather = nugget_scores.gather(1, indices_ascending)
        return nugget_mask, indices_ascending, scores_gather

        # # Step 3: select semantic repr based on nuggets indices
        # selected_semantic_repr = semantic_repr[torch.arange(semantic_repr.shape[0], device=semantic_repr.device).unsqueeze(1), indices_ascending]
        # return selected_semantic_repr, nugget_mask, indices_ascending, scores_gather

    @staticmethod
    def get_valid_mask_indices(mask: torch.Tensor, padding_value: int = -1) -> torch.Tensor:
        """ helper function for ctc indice selection """
        assert (
                len(mask.shape) <= 2
        ), f"Only supports 1D or 2D tensors - got {len(mask.shape)}."
        max_valid = mask.sum(dim=-1).max().item()
        n_nugget = mask.sum(dim=-1)
        # Uses the stable flag to ensure their original order is preserved
        sorted_mask, sorted_indices = torch.sort(
            mask.float(), dim=-1, descending=True, stable=True
        )
        valid_indices = (
            sorted_indices[:, :max_valid]
            if len(mask.shape) == 2
            else sorted_indices[:max_valid]
        )
        valid_indices[
            ~(
                sorted_mask.bool()[:, :max_valid]
                if len(mask.shape) == 2
                else sorted_mask.bool()[:max_valid]
            )
        ] = padding_value
        return valid_indices, max_valid, n_nugget

    def apply_ctc_as_compressor(self, ctc_logits, semantic_repr):
        # nugget_scores = ctc_logits[:, :, self.ctc_info["word_delimiter_token_id"]]
        ctc_pred = torch.argmax(ctc_logits, dim=-1)
        mask = ctc_pred == self.ctc_info["word_delimiter_token_id"]
        indices, max_nugget, n_nugget = self.get_valid_mask_indices(mask, padding_value=-1)
        # then we construct nugget mask using indices, where -1 indicate the repr should not be used
        selected_repr = semantic_repr[torch.arange(semantic_repr.shape[0], device=semantic_repr.device).unsqueeze(1), indices]
        nugget_mask = indices != -1
        return selected_repr, nugget_mask, indices, n_nugget


    @staticmethod
    def compute_regularization_loss(word_delim_logits, nugget_scores, compress_attn_mask):
        # first mask the position we don't see with -inf

        word_delim_logits[~compress_attn_mask] = torch.finfo(word_delim_logits.dtype).min
        nugget_scores[~compress_attn_mask] = torch.finfo(nugget_scores.dtype).min
        target_dist = F.log_softmax(word_delim_logits, dim=-1)
        source_dist = F.log_softmax(nugget_scores, dim=-1)
        kl_loss = F.kl_div(source_dist, target_dist, reduction="none", log_target=True)
        kl_loss = torch.sum(kl_loss * compress_attn_mask, dim=1)
        kl_loss = torch.mean(kl_loss)

        # compute entropy as regularization
        entropy = torch.exp(source_dist) * source_dist
        entropy = torch.sum(entropy * compress_attn_mask, dim=1)
        # here we intentionally use entropy, not neg entropy, as regularization
        entropy_loss = torch.mean(entropy)
        return kl_loss, entropy_loss

    def compute_consecutive_regularization(self, nugget_scores, compress_attn_mask):
        window_size = 5
        reg_loss = 0
        for i in range(1, window_size+1):
            diff = nugget_scores[:, i:] - nugget_scores[:, :-i]
            diff = diff * compress_attn_mask[:, i:]
            reg_loss += torch.mean(diff ** 2)
        # print(-reg_loss)
        return -reg_loss

    @staticmethod
    def add_nugget_type_embedding(inputs_embed, nugget_indices, nugget_mask):
        bz, seq_len, hid_dim = inputs_embed.shape
        nugget_position = nugget_indices.clone()
        nugget_position[~nugget_mask] = -1 # take care of in apply nuggets step
        type_ind_table = inputs_embed.new_full((bz, seq_len + 1), 0).long()
        type_ind_table[torch.arange(bz, device=inputs_embed.device).unsqueeze(1), nugget_position] = 1
        return type_ind_table[:, :-1], nugget_position

    def lm_nugget_training_with_ctc(
            self, audio_seq, audio_attn_mask, lm_text_seq, lm_attn_mask, lm_labels, ctc_labels, misc_info,
            inference=False, is_training_nuggets=False, is_training_ctc_only=False
    ):
        encoder_out, compress_attn_mask = self.encoder(audio_seq, attention_mask=audio_attn_mask)
        ctc_out = self.ctc_head(
            encoder_hidden_states=encoder_out.last_hidden_state,
            encoder_attention_mask=compress_attn_mask,
            labels=ctc_labels,
        )
        # # debug purpose
        # is_training_nuggets=True
        # is_training_ctc_only=False
        if is_training_ctc_only:
            # only pre-train ctc logits
            if inference:
                ctc_gen = torch.argmax(ctc_out.logits, dim=-1)
                return None, ctc_gen, misc_info
            ctc_loss = ctc_out.loss
            misc_info["ctc_loss"] = ctc_loss.item()
            return ctc_loss, misc_info
        elif is_training_nuggets:
            # ctc is pre-trained and frozen, we start updating nuggets with ctc and decoder feedback
            ctc_logits = ctc_out.logits # bz x seq_len x vocab_size (32)
            # KL and entropy loss between current nuggets' distribution and ctc logits
            word_delim_logits = ctc_logits[:, :, self.ctc_info["word_delimiter_token_id"]]
            nugget_scores = self.nugget_model(encoder_out.last_hidden_state).squeeze(-1)
            # not helpful, disable for now
            # kl_loss, entropy_loss = self.compute_regularization_loss(word_delim_logits, nugget_scores, compress_attn_mask)
            consecutive_reg = self.compute_consecutive_regularization(nugget_scores, compress_attn_mask)
            # misc_info["nugget_kl_loss"] = kl_loss.item()
            # misc_info["nugget_entropy_loss"] = entropy_loss.item()
            misc_info["nugget_consecutive_loss"] = consecutive_reg.item()
            kl_loss, entropy_loss = 0, 0
            if self.dynamic_compress_rate:
                # use ctc logits to guide selection of nugget number
                ctc_pred_labels = torch.argmax(ctc_logits, dim=-1)
                word_delimiter_mask = ctc_pred_labels == self.ctc_info["word_delimiter_token_id"]
                # get the increasing ordered indices that has word delim id
                valid_indices, max_valid, n_nugget = self.get_valid_mask_indices(word_delimiter_mask, padding_value=-1)

                consecutive_mask = valid_indices[:, 1:] - valid_indices[:, :-1] != 1
                bz = consecutive_mask.shape[0]
                consecutive_mask = torch.cat([consecutive_mask.new_full((bz, 1), 1).bool(), consecutive_mask], dim=1)
                valid_indices[~consecutive_mask] = -1
                num_word_delim = (valid_indices > 0).sum(dim=1) # bz
                num_word_delim = torch.floor(num_word_delim * self.word2subword_ratio).long()

                # sometimes ctc is wrong or training data is noisy s.t. no delimiter is predicted
                # so we need to cap the min number of selection to a threshold
                # bounded by compression rate 20 for now, and +1 to avoid 0
                min_nuggets_by_threshold = 1 + (compress_attn_mask.sum(dim=1) // 20)
                num_word_delim = torch.max(num_word_delim, min_nuggets_by_threshold)
                ratio = num_word_delim / (compress_attn_mask.sum(dim=1) + 1e-6)
                misc_info["ratio"] = ratio.mean().item()
                # use calculated number of nuggets to select nuggets so the ratio is dynamic
                nugget_mask, nugget_indices, select_nugget_scores = self.apply_nugget_compression(encoder_out.last_hidden_state, compress_attn_mask, n_nugget=num_word_delim, nugget_scores=nugget_scores)
                type_ind_table, nugget_index_to_save = self.add_nugget_type_embedding(encoder_out.last_hidden_state, nugget_indices, nugget_mask)
                inputs_embed = encoder_out.last_hidden_state + self.semantic_encoder.type_embed(type_ind_table)
                semantic_repr = self.semantic_encoder(
                    inputs_embeds=inputs_embed,
                    attention_mask=compress_attn_mask
                ).last_hidden_state
                semantic_repr = semantic_repr[
                    torch.arange(semantic_repr.shape[0], device=semantic_repr.device).unsqueeze(1), nugget_indices]

                semantic_attn_mask = nugget_mask
                misc_info["nugget_indices"] = nugget_index_to_save
                nugget_scores = select_nugget_scores
            else:
                # use fixed compress rate defined by user
                nugget_mask, nugget_indices, select_nugget_scores = self.apply_nugget_compression(
                    encoder_out.last_hidden_state, compress_attn_mask, nugget_scores)
                # add token type embedding to selected indices, index to save has -1
                type_ind_table, nugget_index_to_save = self.add_nugget_type_embedding(encoder_out.last_hidden_state, nugget_indices, nugget_mask)
                inputs_embed = encoder_out.last_hidden_state + self.semantic_encoder.type_embed(type_ind_table)
                semantic_repr = self.semantic_encoder(
                    inputs_embeds=inputs_embed,
                    attention_mask=compress_attn_mask
                ).last_hidden_state

                semantic_repr = semantic_repr[torch.arange(semantic_repr.shape[0], device=semantic_repr.device).unsqueeze(1), nugget_indices]
                semantic_attn_mask = nugget_mask

                misc_info["nugget_indices"] = nugget_index_to_save
                # TODO: delete me! this is just to check if scorer feedback from decoder is harmful when regularizer is used
                nugget_scores = None
            # now start decoding with the compressed repr
            if not inference:
                lm_loss = self.lm_decoder(
                    input_ids=lm_text_seq,
                    attention_mask=lm_attn_mask,
                    encoder_hidden_states=semantic_repr,
                    encoder_attention_mask=semantic_attn_mask,
                    scorer_logits=nugget_scores,
                    labels=lm_labels,
                ).loss
                misc_info["lm_loss"] = lm_loss.item()
                # aggregate regularization loss into lm loss with some scale
                loss = lm_loss + 1e-6 * consecutive_reg
                return loss, misc_info
            else:
                lm_gen = self.lm_generate(semantic_repr, semantic_attn_mask, nugget_scores)
                return lm_gen, None, misc_info
        else:
            # only update lm with frozen scorer
            if self.dynamic_compress_rate:
                ctc_logits = ctc_out.logits  # bz x seq_len x vocab_size (32)
                ctc_pred_labels = torch.argmax(ctc_logits, dim=-1)
                word_delimiter_mask = ctc_pred_labels == self.ctc_info["word_delimiter_token_id"]
                # get the increasing ordered indices that has word delim id
                valid_indices, max_valid, n_nugget = self.get_valid_mask_indices(word_delimiter_mask, padding_value=-1)

                consecutive_mask = valid_indices[:, 1:] - valid_indices[:, :-1] != 1
                bz = consecutive_mask.shape[0]
                consecutive_mask = torch.cat([consecutive_mask.new_full((bz, 1), 1).bool(), consecutive_mask], dim=1)
                if valid_indices.shape[1] > 0:
                    valid_indices[~consecutive_mask] = -1
                num_word_delim = (valid_indices > 0).sum(dim=1)  # bz
                num_word_delim = torch.floor(num_word_delim * self.word2subword_ratio).long()
                min_nuggets_by_threshold = compress_attn_mask.sum(dim=1) // 20
                num_word_delim = torch.max(num_word_delim, min_nuggets_by_threshold)
                ratio = num_word_delim / (compress_attn_mask.sum(dim=1) + 1e-6)
                misc_info["ratio"] = ratio.mean().item()

                # use calculated number of nuggets to select nuggets so the ratio is dynamic
                nugget_mask, nugget_indices, select_nugget_scores = self.apply_nugget_compression(
                    encoder_out.last_hidden_state, compress_attn_mask, n_nugget=num_word_delim)

                type_ind_table, nugget_index_to_save = self.add_nugget_type_embedding(encoder_out.last_hidden_state, nugget_indices,
                                                                nugget_mask)
                inputs_embed = encoder_out.last_hidden_state + self.semantic_encoder.type_embed(type_ind_table)
                semantic_repr = self.semantic_encoder(
                    inputs_embeds=inputs_embed,
                    attention_mask=compress_attn_mask
                ).last_hidden_state
                semantic_repr = semantic_repr[
                    torch.arange(semantic_repr.shape[0], device=semantic_repr.device).unsqueeze(1), nugget_indices]
                semantic_attn_mask = nugget_mask
                misc_info["nugget_indices"] = nugget_index_to_save
                nugget_scores = select_nugget_scores
            else:
                # use fixed compress rate defined by user
                nugget_mask, nugget_indices, select_nugget_scores = self.apply_nugget_compression(
                    encoder_out.last_hidden_state, compress_attn_mask)
                # add token type embedding to selected indices
                type_ind_table, nugget_index_to_save = self.add_nugget_type_embedding(encoder_out.last_hidden_state, nugget_indices,
                                                                nugget_mask)
                inputs_embed = encoder_out.last_hidden_state + self.semantic_encoder.type_embed(type_ind_table)
                semantic_repr = self.semantic_encoder(
                    inputs_embeds=inputs_embed,
                    attention_mask=compress_attn_mask
                ).last_hidden_state
                semantic_repr = semantic_repr[
                    torch.arange(semantic_repr.shape[0], device=semantic_repr.device).unsqueeze(1), nugget_indices]
                semantic_attn_mask = nugget_mask
                misc_info["nugget_indices"] = nugget_index_to_save
                # TODO: delete me
                nugget_scores=None
            if not inference:
                lm_loss = self.lm_decoder(
                    input_ids=lm_text_seq,
                    attention_mask=lm_attn_mask,
                    encoder_hidden_states=semantic_repr,
                    encoder_attention_mask=semantic_attn_mask,
                    scorer_logits=nugget_scores,
                    labels=lm_labels,
                ).loss
                misc_info["lm_loss"] = lm_loss.item()
                return lm_loss, misc_info
            else:
                lm_gen = self.lm_generate(semantic_repr, semantic_attn_mask, nugget_scores)
                return lm_gen, None, misc_info

    def lm_ctc_training(self, audio_seq, audio_attn_mask, lm_text_seq, lm_attn_mask, lm_labels, ctc_labels, misc_info, inference=False):
        encoder_out, compress_attn_mask = self.encoder(audio_seq, attention_mask=audio_attn_mask)
        ctc_out = self.ctc_head(
            encoder_hidden_states=encoder_out.last_hidden_state,
            encoder_attention_mask=compress_attn_mask,
            labels=ctc_labels,
        )
        semantic_repr = self.semantic_encoder(
            inputs_embeds=encoder_out.last_hidden_state,
            attention_mask=compress_attn_mask
        ).last_hidden_state
        if not inference:
            lm_loss = self.lm_decoder(
                input_ids=lm_text_seq,
                attention_mask=lm_attn_mask,
                encoder_hidden_states=semantic_repr,
                encoder_attention_mask=compress_attn_mask,
                labels=lm_labels,
            ).loss
            misc_info["lm_loss"] = lm_loss.item()
            ctc_loss = ctc_out.loss
            misc_info["ctc_loss"] = ctc_loss.item()
            loss = lm_loss + ctc_loss
            return loss, misc_info
        else:
            ctc_gen = torch.argmax(ctc_out.logits, dim=-1)
            lm_gen = self.lm_generate(semantic_repr, compress_attn_mask)
            return lm_gen, ctc_gen, misc_info

    def lm_nugget_training(self, audio_seq, audio_attn_mask, lm_text_seq, lm_attn_mask, lm_labels, misc_info, inference=False):
        encoder_out, compress_attn_mask = self.encoder(audio_seq, attention_mask=audio_attn_mask)
        nugget_scores, consecutive_reg = None, 0
        # apply nuggets on semantic repr
        if self.disable_nuggets:
            bz, seq_len = audio_seq.shape
            if seq_len // (320 * self.nugget_compress_rate) < 10:
                # need to pad the output otherwise the convolution will have padding error
                min_len = self.nugget_compress_rate * 320 * 10 # 320 is w2v compression rate
                padded_audio_seq = audio_seq.new_full((bz, min_len), 0)
                padded_audio_seq[:, :seq_len] = audio_seq
                padded_attn_mask = audio_attn_mask.new_full((bz, min_len), 0)
                padded_attn_mask[:, :seq_len] = audio_attn_mask
                # redo encodeing
                encoder_out, compress_attn_mask = self.encoder(padded_audio_seq, attention_mask=padded_attn_mask)
            # conv based nuggets: directly compress encoder output
            # this is because compress on semantic encoder performs really bad
            semantic_repr, semantic_attn_mask = self.apply_conv_compression(
                encoder_out.last_hidden_state, compress_attn_mask)
            # overwrite the semantic encoder output
            semantic_repr = self.semantic_encoder(
                inputs_embeds=semantic_repr,
                attention_mask=semantic_attn_mask
            ).last_hidden_state
        else:
            full_nugget_scores = self.nugget_model(encoder_out.last_hidden_state).squeeze(-1)
            # apply fixed compress rate nuggets
            nugget_mask, nugget_indices, nugget_scores = self.apply_nugget_compression(
                encoder_out.last_hidden_state, compress_attn_mask, nugget_scores=full_nugget_scores)
            consecutive_reg = self.compute_consecutive_regularization(full_nugget_scores, compress_attn_mask)
            misc_info["nugget_consecutive_loss"] = consecutive_reg.item()
            type_ind_table, nugget_index_to_save = self.add_nugget_type_embedding(encoder_out.last_hidden_state, nugget_indices, nugget_mask)
            inputs_embed = encoder_out.last_hidden_state + self.semantic_encoder.type_embed(type_ind_table)
            misc_info["nugget_indices"] = nugget_index_to_save
            semantic_repr = self.semantic_encoder(
                inputs_embeds=inputs_embed,
                attention_mask=compress_attn_mask
            ).last_hidden_state
            semantic_repr = semantic_repr[
                torch.arange(semantic_repr.shape[0], device=semantic_repr.device).unsqueeze(1), nugget_indices]
            semantic_attn_mask = nugget_mask
        # now decode with the compressed repr
        if not inference:
            lm_loss = self.lm_decoder(
                input_ids=lm_text_seq,
                attention_mask=lm_attn_mask,
                encoder_hidden_states=semantic_repr,
                encoder_attention_mask=semantic_attn_mask,
                scorer_logits=nugget_scores,
                labels=lm_labels,
            ).loss
            misc_info["lm_loss"] = lm_loss.item()
            loss = lm_loss + 1e-6 * consecutive_reg
            loss = lm_loss
            return loss, misc_info
        else:
            lm_gen = self.lm_generate(semantic_repr, semantic_attn_mask, nugget_scores)
            return lm_gen, None, misc_info

    def lm_only_training(self, audio_seq, audio_attn_mask, lm_text_seq, lm_attn_mask, lm_labels, misc_info, inference=False):
        encoder_out, compress_attn_mask = self.encoder(audio_seq, attention_mask=audio_attn_mask)

        semantic_repr = self.semantic_encoder(
            inputs_embeds=encoder_out.last_hidden_state,
            attention_mask=compress_attn_mask
        ).last_hidden_state
        if not inference:
            lm_loss = self.lm_decoder(
                input_ids=lm_text_seq,
                attention_mask=lm_attn_mask,
                encoder_hidden_states=semantic_repr,
                encoder_attention_mask=compress_attn_mask,
                labels=lm_labels,
            ).loss
            misc_info["lm_loss"] = lm_loss.item()
            return lm_loss, misc_info
        else:
            lm_gen = self.lm_generate(semantic_repr, compress_attn_mask)
            return lm_gen, None, misc_info

    def ctc_only_training(self, audio_seq, audio_attn_mask, ctc_labels, misc_info, inference=False):
        encoder_out, compress_attn_mask = self.encoder(audio_seq, attention_mask=audio_attn_mask)
        ctc_out = self.ctc_head(
            encoder_hidden_states=encoder_out.last_hidden_state,
            encoder_attention_mask=compress_attn_mask,
            labels=ctc_labels,
        )
        if not inference:
            ctc_loss = ctc_out.loss
            misc_info["ctc_loss"] = ctc_loss.item()
            return ctc_loss, misc_info
        else:
            ctc_gen = torch.argmax(ctc_out.logits, dim=-1)
            return None, ctc_gen, misc_info

    def resize(self, alphas, target_lengths, noise=0.0, threshold=1.0):
        """
        alpha in thresh=1.0 | (0.0, +0.21)
        target_lengths: if None, apply round and resize, else apply scaling
        """
        device = alphas.device
        # sum
        _num = alphas.sum(-1)

        num = target_lengths.float()
        num = num + noise * torch.rand(alphas.size(0)).to(device)
        # scaling
        _alphas = alphas * (num / _num)[:, None].repeat(1, alphas.size(1))

        # rm attention value that exceeds threashold
        count = 0
        while len(torch.where(_alphas > threshold)[0]):
            count += 1
            if count > 10:
                break
            # print('fixing alpha')
            xs, ys = torch.where(_alphas > threshold)
            # print(xs, ys)
            for x, y in zip(xs, ys):
                if _alphas[x][y] >= threshold:
                    mask = _alphas[x].ne(0).float()
                    mean = 0.5 * _alphas[x].sum() / mask.sum()
                    # print(_alphas[x])
                    _alphas[x] = _alphas[x] * 0.5 + mean * mask
                    # print(_alphas[x])
                    # exit(0)
        return _alphas

    def lm_cif_training(self, audio_seq, audio_attn_mask, lm_text_seq, lm_attn_mask, lm_labels, misc_info, inference=False):
        # code adapted from https://github.com/MingLunHan/CIF-PyTorch/blob/main/modules/cif_middleware.py and https://github.com/dqqcasia/mosst/blob/master/fairseq/models/speech_to_text/convtransformer_wav2vec_cif.py
        device = audio_seq.device
        encoder_out, compress_attn_mask = self.encoder(audio_seq, attention_mask=audio_attn_mask)
        B, T, H = encoder_out.last_hidden_state.shape
        semantic_repr = self.semantic_encoder(
            inputs_embeds=encoder_out.last_hidden_state,
            attention_mask=compress_attn_mask
        ).last_hidden_state
        # bz x seq x 1
        nugget_score = self.nugget_model(encoder_out.last_hidden_state).squeeze(-1)
        # for compression, we do not use threshold, but directly find the value that select top k values(chosen based on compress rate)
        # nugget_mask, indices_ascending, scores_gather = self.apply_nugget_compression(encoder_out.last_hidden_state, compress_attn_mask, nugget_scores=nugget_score)
        # bz x n_nugget
        # indices_ascending[~nugget_mask] = -1
        # ind_table = audio_seq.new_full((B, T + 1), 0).long()
        # ind_table[torch.arange(B, device=device).unsqueeze(1), indices_ascending] = 1
        # ind_table = ind_table[:, :-1]

        alphas = torch.sigmoid(nugget_score)
        alphas = alphas * compress_attn_mask
        target_len = (compress_attn_mask.sum(dim=1) * (1 /self.nugget_compress_rate) + 0.99).to(torch.int64)
        threshold = 1.0
        alphas = self.resize(alphas, target_lengths=target_len, noise=0.0, threshold=threshold)

        integrate = torch.zeros([B], device=device)
        frame = torch.zeros([B, H], device=device)
        list_fires, list_frames = [], []
        list_fired_positions = []
        for t in range(T):
            # perform integrate and fire
            alpha = alphas[:, t]
            distribution_completion = torch.ones([B], device=device) - integrate
            # print(distribution_completion)
            integrate += alpha
            list_fires.append(integrate)
            fire_place = integrate >= threshold
            # fire_place = ind_table[:, t] == 1
            list_fired_positions.append(fire_place)
            integrate = torch.where(fire_place,
                                    integrate - torch.ones([B], device=device),
                                    integrate)
            cur = torch.where(fire_place,
                              distribution_completion,
                              alpha)
            remainds = alpha - cur

            frame += cur[:, None] * semantic_repr[:, t, :]
            list_frames.append(frame)
            frame = torch.where(fire_place[:, None].repeat(1, H),
                                remainds[:, None] * semantic_repr[:, t, :],
                                frame)
        # aggreagte results
        fires = torch.stack(list_fires, 1)
        frames = torch.stack(list_frames, 1)
        fired_positions = torch.stack(list_fired_positions, 1)
        num_fires = fired_positions.sum(dim=1)
        ratio = num_fires / (compress_attn_mask.sum(dim=1) + 1e-6)
        misc_info["ratio"] = ratio.mean().item()

        list_ls = []
        max_label_len = num_fires.max().item()
        cif_attn_mask = torch.zeros([B, max_label_len], device=device)
        for b in range(B):
            fired_position = fired_positions[b, :]
            l = torch.index_select(frames[b, :, :], 0, torch.where(fired_position)[0])
            pad_l = torch.zeros([max_label_len - l.size(0), H], device=device)
            list_ls.append(torch.cat([l, pad_l], 0))
            cif_attn_mask[b, :l.size(0)] = 1
        cif_repr = torch.stack(list_ls, 0)
        # now we operate on cif repr and mask for
        valid_indices, _, _ = self.get_valid_mask_indices(fired_positions, padding_value=-1)
        misc_info["nugget_indices"] = valid_indices
        if not inference:
            lm_loss = self.lm_decoder(
                input_ids=lm_text_seq,
                attention_mask=lm_attn_mask,
                encoder_hidden_states=cif_repr,
                encoder_attention_mask=cif_attn_mask,
                labels=lm_labels,
            ).loss
            misc_info["lm_loss"] = lm_loss.item()
            return lm_loss, misc_info
        else:
            lm_gen = self.lm_generate(cif_repr, cif_attn_mask)
            return lm_gen, None, misc_info

    def forward(
            self,
            audio_seq,
            audio_attn_mask,
            lm_text_seq,
            lm_attn_mask,
            lm_labels,
            ctc_text_seq,
            ctc_attn_mask,
            ctc_labels,
            inference=False,
            is_training_nuggets=False,
            is_training_ctc_only=False,
    ):
        # route dispatcher to different training setup
        misc_info = dict()
        # prepare some general helpful info that might be added
        for keys in ["nugget_indices", "ratio", "nugget_kl_loss", "nugget_entropy_loss"]:
            misc_info[keys] = None

        if self.use_lm:
            if self.use_ctc:
                if self.use_nuggets:
                    assert self.use_ctc_as_scorer, "when nugget and ctc are used, ctc are only used as regularization"
                    return self.lm_nugget_training_with_ctc(
                        audio_seq, audio_attn_mask, lm_text_seq, lm_attn_mask, lm_labels, ctc_labels, misc_info,
                        inference=inference, is_training_nuggets=is_training_nuggets, is_training_ctc_only=is_training_ctc_only)
                else:
                    # jointly train lm + ctc
                    return self.lm_ctc_training(audio_seq, audio_attn_mask, lm_text_seq, lm_attn_mask, lm_labels, ctc_labels, misc_info, inference=inference)
            else:
                if self.use_nuggets:
                    # nugget to compress lm, nugget could be conv/regular/cif
                    if self.use_cif:
                        return self.lm_cif_training(audio_seq, audio_attn_mask, lm_text_seq, lm_attn_mask, lm_labels, misc_info, inference=inference)
                    else:
                        return self.lm_nugget_training(audio_seq, audio_attn_mask, lm_text_seq, lm_attn_mask, lm_labels, misc_info, inference=inference)
                else:
                    return self.lm_only_training(audio_seq, audio_attn_mask, lm_text_seq, lm_attn_mask, lm_labels, misc_info, inference=inference)
        else:
            # only train ctc as sanity check for ctc performance
            return self.ctc_only_training(audio_seq, audio_attn_mask, ctc_labels, misc_info, inference=inference)

    def lm_generate(self, semantic_repr, semantic_attn_mask, nugget_scores=None):
        bz = semantic_repr.shape[0]
        bos_token = semantic_repr.new_full((bz, 1), self.lm_info["bos_token_id"]).long()
        generation_config = GenerationConfig(
            max_new_tokens=256,
            do_sample=False,
            bos_token_id=self.lm_info["bos_token_id"],
            eos_token_id=self.lm_info["eos_token_id"],
            pad_token_id=self.lm_info["pad_token_id"],
        )
        # beam search config
        # generation_config = GenerationConfig(
        #     max_new_tokens=128,
        #     num_beams=5,
        #     remove_invalid_values=True,
        #     bos_token_id=self.lm_info["bos_token_id"],
        #     eos_token_id=self.lm_info["eos_token_id"],
        #     pad_token_id=self.lm_info["pad_token_id"],
        # )
        lm_gen = self.lm_decoder.generate(
            generation_config=generation_config,
            input_ids=bos_token,
            encoder_hidden_states=semantic_repr,
            encoder_attention_mask=semantic_attn_mask,
            scorer_logits=nugget_scores,
        )
        return lm_gen
