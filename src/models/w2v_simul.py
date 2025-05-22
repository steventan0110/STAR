import functools
from functools import partial
import torch
import torch.nn.functional as F
from einops import rearrange, unpack, pack
from torch import nn
from torch.linalg import vector_norm
from torchaudio.functional import resample
from transformers import GPT2Config, GenerationConfig
from src.models.my_wav2vec import MyWav2Vec2Model, MyWav2Vec2ForCTC, MyCausalWav2Vec2Model
from src.models.my_transformer import MyGPT2ForLM, MyGPT2Model
from src.models.utils import ResidualUnit
from transformers.models.llama.modeling_llama import LlamaRMSNorm
import math

def exists(val):
    return val is not None

def round_down_nearest_multiple(num, divisor):
    return num // divisor * divisor


def curtail_to_multiple(t, mult, from_left=False):
    data_len = t.shape[-1]
    rounded_seq_len = round_down_nearest_multiple(data_len, mult)
    seq_slice = slice(None, rounded_seq_len) if not from_left else slice(-rounded_seq_len, None)
    return t[..., seq_slice]


class BlockAttnLayer(nn.Module):
    def __init__(self, hidden_size, n_head):
        super(BlockAttnLayer, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head

        self.head_dim = self.hidden_size // self.n_head
        self.q_proj = nn.Linear(self.hidden_size, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_head * self.head_dim, self.hidden_size, bias=False)

        self.input_layernorm = LlamaRMSNorm(self.hidden_size, eps=1e-6)
        self.post_attention_layernorm = LlamaRMSNorm(self.hidden_size, eps=1e-6)

        self.intermediate_size = self.hidden_size * 4
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, semantic_repr, attn_mask):
        bsz, seq_len, _ = semantic_repr.shape  # convnet and tgt has the same length
        residual = semantic_repr
        semantic_repr = self.input_layernorm(semantic_repr)
        query_states = self.q_proj(semantic_repr)
        key_states = self.k_proj(semantic_repr)
        value_states = self.v_proj(semantic_repr)
        # separaet into heads
        query_states = query_states.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1,2)
        key_states = key_states.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        qk_value = torch.matmul(query_states, key_states.transpose(2, 3))
        # bz x num_head x seq_len x seq_len
        attn_weights = qk_value / math.sqrt(self.head_dim)
        # add the preprocessed blockwise attention, attn mask has shape bz x seq_len x seq_len
        attn_mask = attn_mask.float()
        attn_mask[attn_mask == 0] = torch.finfo(attn_weights.dtype).min
        attn_mask = attn_mask.unsqueeze(1) # for num head
        attn_weights = attn_weights + attn_mask


        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        assert attn_output.size() == (bsz, self.n_head, seq_len, self.head_dim)
        # bz, #head, seq_len, head_dim -> bz, seq_len, #head, head_dim -> bz, seq_len, hidden_dim
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        output_states = attn_output + residual
        residual = output_states
        output_states = self.post_attention_layernorm(output_states)
        output_states = self.down_proj(self.act_fn(self.gate_proj(output_states)) * self.up_proj(output_states))
        output_states = residual + output_states
        return output_states


class W2VSimulModel(nn.Module):
    def __init__(
            self,
            lm_info=None,
            ctc_info=None,
            hidden_size=512,
            target_sample_hz=16000,
            use_nuggets=False,
            use_cif=False,
            nugget_compress_rate=12,
            cache_dir=None,
            use_ilk=False,
            max_position=2048,
    ):
        super().__init__()
        self.lm_info = lm_info
        self.ctc_info = ctc_info
        self.target_sample_hz = target_sample_hz  # for resampling on the fly
        self.use_nuggets = use_nuggets
        self.use_cif = use_cif
        self.nugget_model = None
        self.hidden_size = hidden_size
        self.word2subword_ratio = 1.5

        self.encoder = MyCausalWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=cache_dir)
        # still freeze feature extractor
        self.encoder.freeze_feature_extractor()
        # on top of frozen pre-trained wav2vec2 encoder, we add a decoder to extract semantic feature
        semantic_encoder_config = GPT2Config(
            n_embd=self.hidden_size,
            # the special token id and vocab size here does not matter
            # since semantic encoder is only dealing with audio feature
            vocab_size=4,
            num_labels=4,
            n_positions=max_position,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            n_layer=4,
            n_head=8,
            add_cross_attention=False
        )
        self.semantic_encoder = MyGPT2Model(semantic_encoder_config)
        ctc_head_config = self.encoder.config
        # ctc tokenization is different, reset the special token ids
        ctc_head_config.vocab_size = ctc_info["vocab_size"]
        ctc_head_config.num_labels = ctc_info["vocab_size"]
        ctc_head_config.bos_token_id = ctc_info["bos_token_id"]
        ctc_head_config.eos_token_id = ctc_info["eos_token_id"]
        ctc_head_config.pad_token_id = ctc_info["pad_token_id"]
        self.ctc_head = MyWav2Vec2ForCTC(
            ctc_head_config,
        )
        self.limit_cross_attention = use_ilk
        print("Using ILK: ", self.limit_cross_attention)
        if self.use_nuggets or self.use_cif:
            # cif and nugget share the same nugget model to have equal #params
            print(f"Using nugget: {self.use_nuggets}; Using cif: {self.use_cif}")
            # basically just to learn a scorer
            self.nugget_model = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, 1)
            )
            self.block_attn = BlockAttnLayer(self.hidden_size, 8)
        else:
            print("No compression mechanism is used")

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
        # decoder is different from before that it has Infinite-look back attention
        # for the cross attention
        self.lm_decoder = MyGPT2ForLM(lm_decoder_config, limit_cross_attn=self.limit_cross_attention)

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

    def apply_nugget_compression(
            self, encoder_repr, attn_mask, target_len,
            nugget_scores=None, n_nugget=None
    ):
        """
        encoder_repr: (bz, seq_len, hid_dim) from acoustic encoder
        semantic_repr: (bz, seq_len, hid_dim) from semantic encoder
        attn_mask: (bz, seq_len) from both encoder (as they have equal seq len)
        """
        # Step 1: compute nugget scores on acoustic repr (it's frozen repr)
        n_tok = attn_mask.sum(dim=1)
        if n_nugget is None:
            n_nugget = target_len
            # ratio = target_len / n_tok
            # n_nugget = (n_tok * ratio + 0.99).to(torch.int64)
            # n_nugget[n_nugget > n_tok] = n_tok[n_nugget > n_tok]

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


    def apply_sliding_nugget(self, nugget_scores, attn_mask, target_length):
        # compute the latency loss to make nugget score more spread out
        # also make it activate over certain threshold rather than top k
        bsz, seq_len = nugget_scores.shape
        device = nugget_scores.device

        gather_scores, gather_indices = [], []

        interval = (seq_len / target_length)
        min_interval = 5 # at least jump 5 frames, this is to avoid clustering of indices of high info places
        interval = interval.to(torch.int64)
        src_length = attn_mask.sum(dim=1)

        start_indices = nugget_scores.new_full((bsz,), 0)
        mask_gather = []
        # need to handle the index update in batch fasion, similar to batched decoding
        already_eos = torch.zeros((bsz,), device=device).bool()
        while True:
            end_indices = start_indices + interval
            end_indices = torch.min(end_indices, src_length)
            reach_end = end_indices >= src_length
            start_mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len) >= start_indices.unsqueeze(1)
            end_mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len) < end_indices.unsqueeze(1)
            mask = start_mask & end_mask
            score_copy = nugget_scores.clone().detach()
            score_copy[~mask] = torch.finfo(torch.float32).min
            # now do top-1 selection on the mask
            max_values, max_indices = score_copy.max(dim=-1)
            gather_indices.append(max_indices.unsqueeze(1))
            mask_gather.append(already_eos.unsqueeze(1))
            # this is to make already eos lag by one step, so that mask gather can still append the last index
            already_eos = already_eos | reach_end
            # handle eos checking
            if torch.all(reach_end):
                break
            next_indice = torch.min(max_indices + min_interval, src_length - interval)
            start_indices = torch.where(reach_end, start_indices, next_indice)

        # bz, num_selection
        gather_indices = torch.cat(gather_indices, dim=1).long()
        # select scores out
        gather_scores = nugget_scores.gather(1, gather_indices)
        nugget_mask = torch.cat(mask_gather, dim=1)
        nugget_mask = ~nugget_mask

        return nugget_mask, gather_indices, gather_scores


    def apply_fix_window_nugget(self, nugget_scores, attn_mask, target_length):
        # set hard compress rate of 12
        bsz, seq_len = nugget_scores.shape
        device = nugget_scores.device
        src_length = attn_mask.sum(dim=1)
        num_window = target_length
        window_stride = torch.max(src_length.new_full((bsz,), 5), (src_length / num_window).to(torch.int64))
        # print(window_stride)
        # print(window_stride)
        gather_scores, gather_indices = [], []
        mask_gather = []
        # need to handle the index update in batch fasion, similar to batched decoding
        already_eos = torch.zeros((bsz,), device=device).bool()
        start_indices = nugget_scores.new_full((bsz,), 0)

        while True:
            end_indices = start_indices + window_stride
            end_indices = torch.min(end_indices, src_length)
            reach_end = end_indices >= src_length
            start_mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz,
                                                                                  seq_len) >= start_indices.unsqueeze(1)
            end_mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len) < end_indices.unsqueeze(1)
            mask = start_mask & end_mask
            score_copy = nugget_scores.clone().detach()
            score_copy[~mask] = torch.finfo(torch.float32).min
            # now do top-1 selection on the mask
            max_values, max_indices = score_copy.max(dim=-1)
            gather_indices.append(max_indices.unsqueeze(1))
            mask_gather.append(already_eos.unsqueeze(1))
            # this is to make already eos lag by one step, so that mask gather can still append the last index
            already_eos = already_eos | reach_end
            # handle eos checking
            if torch.all(reach_end):
                break
            next_indice = torch.min(end_indices, src_length - window_stride)
            start_indices = torch.where(reach_end, start_indices, next_indice)
        # bz, num_selection
        gather_indices = torch.cat(gather_indices, dim=1).long()
        # select scores out
        gather_scores = nugget_scores.gather(1, gather_indices)
        nugget_mask = torch.cat(mask_gather, dim=1)
        nugget_mask = ~nugget_mask

        return nugget_mask, gather_indices, gather_scores



    def apply_threshold_yield_nugget(self, alphas, attn_mask):
        cur_sum = alphas.new_full((alphas.shape[0],), 0)
        index_mask = []
        alphas = alphas * attn_mask
        src_len = attn_mask.sum(dim=1)
        bz, seq_len = attn_mask.shape
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=alphas.device), diagonal=0).unsqueeze(0).expand(bz, -1, -1).bool()
        block_attn_list = []
        prev_index = alphas.new_full((bz, 1), -1)
        for i in range(alphas.shape[1]):
            cur_sum += alphas[:, i]
            yield_cur_index = cur_sum >= 1
            yield_cur_index = yield_cur_index | (src_len == i + 1)
            cur_sum = torch.where(yield_cur_index, cur_sum - 1, cur_sum)
            index_mask.append(yield_cur_index)
            # bz x seq_len
            cur_block_attn_mask = torch.arange(seq_len, device=alphas.device).unsqueeze(0).expand(bz, -1)
            cur_block_attn_mask = cur_block_attn_mask > prev_index
            block_attn_list.append(cur_block_attn_mask.unsqueeze(1))
            prev_index[yield_cur_index] = i
            # prev_index = torch.where(yield_cur_index, i, prev_index)
        mask = torch.stack(index_mask, dim=1)
        nugget_indices, _, _ = self.get_valid_mask_indices(mask)
        nugget_mask = nugget_indices != -1
        block_attn_mask = torch.stack(block_attn_list, dim=1).squeeze()
        block_attn_mask = (block_attn_mask & causal_mask).float()
        block_attn_mask[attn_mask == 0] = 0
        return nugget_mask, nugget_indices, block_attn_mask



    def lm_nugget_training(self, audio_seq, audio_attn_mask, lm_text_seq, lm_attn_mask, lm_labels, misc_info, inference=False):
        encoder_out, compress_attn_mask = self.encoder(audio_seq, attention_mask=audio_attn_mask)
        target_lengths = lm_attn_mask.sum(dim=1)
        encoder_repr_for_nugget = encoder_out.last_hidden_state.clone().detach()
        full_nugget_scores = self.nugget_model(encoder_repr_for_nugget).squeeze(-1)
        ############ Nugget Selection Logic ################
        """
            several strategies are possible such as 
            1. fixed compress rate
            2. sliding window 
            3. cif style threshold yielding over window
        """
        # apply fixed compress rate nuggets
        # nugget_mask, nugget_indices, nugget_scores = self.apply_nugget_compression(
        #     encoder_repr_for_nugget, compress_attn_mask, target_lengths, nugget_scores=full_nugget_scores)
        # apply sliding window to select nuggets to enforce spread out
        # nugget_mask, nugget_indices, nugget_scores = self.apply_sliding_nugget(full_nugget_scores, compress_attn_mask, target_lengths)
        # nugget_mask, nugget_indices, nugget_scores = self.apply_fix_window_nugget(full_nugget_scores, compress_attn_mask, target_lengths)
        threshold = 1
        alphas = torch.sigmoid(full_nugget_scores)
        alphas = alphas * compress_attn_mask
        quantity_loss = ((alphas / threshold).sum(dim=1) - target_lengths) ** 2
        quantity_loss = quantity_loss.mean()
        misc_info["quantity_loss"] = quantity_loss.item()
        alphas = self.resize(alphas, target_lengths)
        nugget_mask, nugget_indices, block_attn_mask = self.apply_threshold_yield_nugget(alphas.clone().detach(), compress_attn_mask)
        nugget_scores = full_nugget_scores[torch.arange(full_nugget_scores.shape[0], device=full_nugget_scores.device).unsqueeze(1), nugget_indices]



        #### REGULARIZATION ####
        # alphas = torch.sigmoid(full_nugget_scores)
        # # print(alphas[0, :])
        # selected_alphas = alphas[
        #     torch.arange(alphas.shape[0], device=alphas.device).unsqueeze(1), nugget_indices]
        # selected_alphas[~nugget_mask] = 1
        # # for the selected position's score, we want it to be high
        # reg_loss = torch.sum((1 - selected_alphas) ** 2, dim=1)
        # # for the unselected indices, we want them to be low
        # unselected_alphas = alphas.clone()
        # unselected_alphas[
        #     torch.arange(alphas.shape[0], device=alphas.device).unsqueeze(1), nugget_indices] = 0
        # unselected_alphas[~compress_attn_mask] = 0
        # reg_loss += torch.sum(unselected_alphas ** 2, dim=1)
        # reg_loss = reg_loss.mean()
        # misc_info["nugget_reg_loss"] = reg_loss.item()
        ####### REGULARIZATION #######

        ratio = (nugget_mask.sum(dim=1) / compress_attn_mask.sum(dim=1)).mean()
        # print(ratio.item())
        misc_info["ratio"] = ratio.mean().item()
        # print(f"Ratio: {ratio.mean().item()}")
        type_ind_table, nugget_index_to_save = self.add_nugget_type_embedding(encoder_out.last_hidden_state, nugget_indices, nugget_mask)
        inputs_embed = encoder_out.last_hidden_state + self.semantic_encoder.type_embed(type_ind_table)
        misc_info["nugget_indices"] = nugget_index_to_save
        semantic_repr = self.semantic_encoder(
            inputs_embeds=inputs_embed,
            attention_mask=compress_attn_mask
        ).last_hidden_state

        # hard select the semantic repr
        use_soft = False
        if use_soft:
            # transform semantic repr further with block-wise attn to enforce focus on current block
            # can be seen as a form of local attn (or like cif's weighted sum of each activation block)
            semantic_repr = self.block_attn(semantic_repr, block_attn_mask)

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
            # print(reg_loss)
            # loss = lm_loss + 0.01 * reg_loss
            loss = lm_loss + 0.01 * quantity_loss
            return loss, misc_info
        else:
            lm_gen = self.lm_generate(semantic_repr, semantic_attn_mask, nugget_scores)
            return lm_gen, None, misc_info

    def lm_only_training(self, audio_seq, audio_attn_mask, lm_text_seq, lm_attn_mask, lm_labels, ctc_labels, misc_info, inference=False, train_ctc=False):
        encoder_out, compress_attn_mask = self.encoder(audio_seq, attention_mask=audio_attn_mask)
        semantic_repr = self.semantic_encoder(
            inputs_embeds=encoder_out.last_hidden_state,
            attention_mask=compress_attn_mask
        ).last_hidden_state
        # also pre-train ctc module for inference
        if not inference:
            if train_ctc:
                ctc_loss = self.ctc_head(
                    encoder_hidden_states=encoder_out.last_hidden_state,
                    encoder_attention_mask=compress_attn_mask,
                    labels=ctc_labels,
                ).loss
                misc_info["ctc_loss"] = ctc_loss.item()
                return ctc_loss, misc_info
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
            if train_ctc:
                logits = self.ctc_head(
                    encoder_hidden_states=encoder_out.last_hidden_state,
                    encoder_attention_mask=compress_attn_mask,
                    labels=ctc_labels,
                ).logits
                ctc_gen = torch.argmax(logits, dim=-1)
                return None, ctc_gen, misc_info
            lm_gen = self.lm_generate(semantic_repr, compress_attn_mask)
            return lm_gen, None, misc_info

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
        device = audio_seq.device
        encoder_out, compress_attn_mask = self.encoder(audio_seq, attention_mask=audio_attn_mask)

        B, T, H = encoder_out.last_hidden_state.shape
        semantic_repr = self.semantic_encoder(
            inputs_embeds=encoder_out.last_hidden_state,
            attention_mask=compress_attn_mask
        ).last_hidden_state
        # detach the encoder for scorer
        encoder_repr_for_nugget = encoder_out.last_hidden_state.clone().detach()
        nugget_score = self.nugget_model(encoder_repr_for_nugget).squeeze(-1)
        alphas = torch.sigmoid(nugget_score)
        alphas = alphas * compress_attn_mask
        target_len = lm_attn_mask.sum(dim=1)
        threshold = 1.0

        quantity_loss = ((alphas / threshold).sum(dim=1) - target_len) ** 2
        quantity_loss = quantity_loss.mean()

        misc_info["quantity_loss"] = quantity_loss.item()
        alphas = self.resize(alphas, target_lengths=target_len, noise=0.0, threshold=threshold)

        integrate = torch.zeros([B], device=device)
        frame = torch.zeros([B, H], device=device)
        list_fires, list_frames = [], []
        list_fired_positions = []
        # use to track delays for each position to compute DAL for latency training
        list_delays = []
        # delay = torch.zeros([B], device=device)
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
            # delay += cur * t
            frame += cur[:, None] * semantic_repr[:, t, :]
            list_frames.append(frame)
            # list_delays.append(delay)
            frame = torch.where(fire_place[:, None].repeat(1, H),
                                remainds[:, None] * semantic_repr[:, t, :],
                                frame)
            # delay = torch.where(fire_place,
            #                      remainds * t,
            #                      delay)
        # aggreagte results
        fires = torch.stack(list_fires, 1)
        frames = torch.stack(list_frames, 1)
        # delays = torch.stack(list_delays, 1)
        fired_positions = torch.stack(list_fired_positions, 1)
        num_fires = fired_positions.sum(dim=1)
        list_ls = []
        max_label_len = num_fires.max().item()
        cif_attn_mask = torch.zeros([B, max_label_len], device=device)
        dal_list = []
        for b in range(B):
            fired_position = fired_positions[b, :]
            l = torch.index_select(frames[b, :, :], 0, torch.where(fired_position)[0])
            pad_l = torch.zeros([max_label_len - l.size(0), H], device=device)
            list_ls.append(torch.cat([l, pad_l], 0))
            cif_attn_mask[b, :l.size(0)] = 1
            # compute DAL, this might add to much complexity, discard for now
            # delay_b = torch.index_select(delays[b, :], 0, torch.where(fired_position)[0])
            # num_delay = delay_b.shape[0]
            # gamma = num_delay / target_len[b]
            # for i in range(num_delay):
            #     if i == 0:
            #         dal_list.append(delay_b[i])
            #     else:
            #         dal_list.append(max(delay_b[i], dal_list[-1] + gamma))
            # print(delay_b)
            # print(delay_b.shape)
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
            loss = lm_loss + 0.1 * quantity_loss
            return loss, misc_info
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
            is_training_ctc=False
    ):
        # route dispatcher to different training setup
        misc_info = dict()
        # prepare some general helpful info that might be added
        for keys in ["nugget_indices", "ratio", "nugget_kl_loss", "nugget_entropy_loss"]:
            misc_info[keys] = None

        if self.use_cif:
            return self.lm_cif_training(
                audio_seq, audio_attn_mask, lm_text_seq, lm_attn_mask, lm_labels, misc_info, inference=inference)
        elif self.use_nuggets:
            return self.lm_nugget_training(
                audio_seq, audio_attn_mask, lm_text_seq, lm_attn_mask, lm_labels, misc_info, inference=inference)
        else:
            # jointly train lm + ctc
            return self.lm_only_training(audio_seq, audio_attn_mask, lm_text_seq, lm_attn_mask, lm_labels, ctc_labels, misc_info, inference=inference, train_ctc=is_training_ctc)

    def lm_generate(self, semantic_repr, semantic_attn_mask, nugget_scores=None):
        bz = semantic_repr.shape[0]
        bos_token = semantic_repr.new_full((bz, 1), self.lm_info["bos_token_id"]).long()
        generation_config = GenerationConfig(
            max_new_tokens=128,
            do_sample=False,
            bos_token_id=self.lm_info["bos_token_id"],
            eos_token_id=self.lm_info["eos_token_id"],
            pad_token_id=self.lm_info["pad_token_id"],
        )
        lm_gen = self.lm_decoder.generate(
            generation_config=generation_config,
            input_ids=bos_token,
            encoder_hidden_states=semantic_repr,
            encoder_attention_mask=semantic_attn_mask,
            scorer_logits=nugget_scores,
        )
        return lm_gen
