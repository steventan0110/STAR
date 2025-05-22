from typing import Optional, Union, Tuple, List
import torch
import re
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2ForCTC, GPT2Config, Wav2Vec2CTCTokenizer
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Encoder, Wav2Vec2EncoderLayer, Wav2Vec2Attention
from transformers.modeling_outputs import CausalLMOutput, Wav2Vec2BaseModelOutput, BaseModelOutput, BaseModelOutputWithPast
from transformers.tokenization_utils_base import TextInput, AddedToken
from src.models.my_transformer import MyGPT2Model
from tokenizers import Tokenizer


# class MyWav2VecModel()
_HIDDEN_STATES_START_POSITION = 2


class MyWav2Vec2ForCTC(Wav2Vec2ForCTC):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = None # we manually pass the encoder outside

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[bool] = None,
        encoder_attention_mask: Optional[bool] = None,
        input_lengths: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        # use to transform the representation
        hidden_states = self.dropout(encoder_hidden_states)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            if input_lengths is None:
                # retrieve loss input_lengths from attention_mask
                if encoder_attention_mask is not None:
                    input_lengths = encoder_attention_mask.sum(-1).to(torch.long)
                else:
                    attention_mask = (
                        attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
                    )
                    input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
            else:
                input_lengths = input_lengths.to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)
            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            # print(log_probs)
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
            # print(loss)

        return CausalLMOutput(
            loss=loss, logits=logits
        )


class MyWav2Vec2Model(Wav2Vec2Model):
    def forward(
            self,
            input_values: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            mask_time_indices: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # feature extractor expects input of shape (bz, dim) instead of (bz, 1, dim)
        extract_features = self.feature_extractor(input_values.squeeze(1))
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )
        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        ), attention_mask


class MyCausalWav2Vec2Attention(Wav2Vec2Attention):
    # overwrite the forward fnc to use causal attention
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        # print("is_cross_attention: ", is_cross_attention)
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )
        # add triangular mask for causal attention
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len)), diagonal=0)
        if tgt_len != src_len:
            # only possible when past kv is given, other self-attn is always same len
            # preprend 1's to the mask
            causal_mask = torch.cat([torch.ones((tgt_len, src_len - tgt_len)), causal_mask], dim=1)
        causal_mask = causal_mask.to(attn_weights.device).unsqueeze(0).expand(bsz * self.num_heads, -1, -1)
        causal_mask = causal_mask.masked_fill(causal_mask == 0, torch.finfo(attn_weights.dtype).min)
        attn_weights = attn_weights + causal_mask

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

class MyCausalWav2Vec2EncoderLayer(Wav2Vec2EncoderLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = MyCausalWav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True, # this is for simul_st inference time
        )

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, past_key_value=None):
        attn_residual = hidden_states
        hidden_states, attn_weights, present_key_value = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_value,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        # always give past key values
        outputs = (hidden_states, present_key_value)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class MyCausalWav2Vec2Encoder(Wav2Vec2Encoder):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([MyCausalWav2Vec2EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
    ):
        # we make this version support past key values
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        new_past_key_values = ()
        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            ) # bz x 1 x tgt_len x tgt_len
            if past_key_values is not None:
                past_key_values_length = past_key_values[0][0].shape[2]
                # expand attention mask to include past key values,
                attention_mask = torch.cat(
                    [
                        attention_mask.new_ones((attention_mask.shape[0], 1, attention_mask.shape[-1], past_key_values_length)),
                        attention_mask,
                    ],
                    dim=-1,
                )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = False

        for layer_id, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask,
                    )
                else:
                    past_key_value = past_key_values[layer_id] if past_key_values is not None else None
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        output_attentions=output_attentions,
                        past_key_value=past_key_value,
                    )

                hidden_states = layer_outputs[0]
                # the second argument is always keyvalue returned from attn module
                new_past_key_values += (layer_outputs[1],)

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            past_key_values=new_past_key_values,
        )


class MyCausalWav2Vec2Model(Wav2Vec2Model):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = MyCausalWav2Vec2Encoder(config)
    def forward(
            self,
            input_values: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            mask_time_indices: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            past_key_values: Optional[Tuple[torch.Tensor]] = None,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # feature extractor expects input of shape (bz, dim) instead of (bz, 1, dim)
        extract_features = self.feature_extractor(input_values.squeeze(1))
        extract_features = extract_features.transpose(1, 2)
        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        # this is only excuted in training time
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )
        # use our own causal encoder
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            past_key_values=encoder_outputs.past_key_values,
        ), attention_mask

        # return Wav2Vec2BaseModelOutput(
        #     last_hidden_state=hidden_states,
        #     extract_features=extract_features,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        #     past_key_values=encoder_outputs.past_key_values,
        # ), attention_mask


class MyWav2Vec2CTCTokenizer(Wav2Vec2CTCTokenizer):
    def __init__(self, vocab_file, unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|", **kwargs):
        super().__init__(vocab_file=vocab_file, unk_token=unk_token, pad_token=pad_token, word_delimiter_token=word_delimiter_token, **kwargs)

    def tokenize(self, text: TextInput, **kwargs) -> List[str]:

        """
        Converts a string in a sequence of tokens, using the tokenizer.

        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces). Takes care of added tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            **kwargs (additional keyword arguments):
                Passed along to the model-specific `prepare_for_tokenization` preprocessing method.

        Returns:
            `List[str]`: The list of tokens.
        """
        split_special_tokens = kwargs.pop("split_special_tokens", self.split_special_tokens)

        text, kwargs = self.prepare_for_tokenization(text, **kwargs)


        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase
            escaped_special_toks = [re.escape(s_tok) for s_tok in (self.all_special_tokens)]
            escaped_special_toks += [
                re.escape(s_tok.content)
                for s_tok in (self._added_tokens_decoder.values())
                if not s_tok.special and s_tok.normalized
            ]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

        if split_special_tokens:
            no_split_token = []
            tokens = [text]
        else:
            no_split_token = set(self._added_tokens_encoder.keys())  # don't split on any of the added tokens
            tokens = self.tokens_trie.split(text)

        tokenized_text = []
        for token in tokens:
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token))
        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text


class MyLMTokenizer:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.bos_token_id = self.tokenizer.token_to_id("<s>")
        self.eos_token_id = self.tokenizer.token_to_id("</s>")
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.unk_token_id = self.tokenizer.token_to_id("<unk>")

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def __call__(self, text: str, **kwargs):
        encoding = self.tokenizer.encode(text)
        ids = torch.tensor(encoding.ids, dtype=torch.long)
        output_dict = {
            "input_ids": ids.unsqueeze(0),
        }
        return output_dict

    def decode(self, ids, **kwargs):
        ids = ids.tolist()
        return self.tokenizer.decode(ids)

    def batch_decode(self, ids, **kwargs):
        return [self.decode(ids_, **kwargs) for ids_ in ids]
