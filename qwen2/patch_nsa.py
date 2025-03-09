import importlib
from nsa_attention.nsa_attn import NsaAttention
from triton_kernel.fused_apply_rope import fused_apply_rope


from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Union, Tuple, List
import torch
import torch.nn as nn


module = importlib.import_module('transformers.models.qwen2.modeling_qwen2')


class Qwen2NSA(torch.nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.qk_head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.qk_head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.v_head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.nsa = NsaAttention(qk_head_dim=self.head_dim, 
                                v_head_dim=self.head_dim, 
                                kernel_size=config.kernel_size,
                                stride=config.stride,
                                select_size=config.select_size,
                                top_n=config.top_n,
                                window_size=config.window_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bs, seq_len, d = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(bs, seq_len, -1, self.qk_head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bs, seq_len, -1, self.qk_head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bs, seq_len, -1, self.v_head_dim)

        cos, sin = position_embeddings
        query_states, key_states = fused_apply_rope(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        attn_output = self.nsa(query_states, key_states, value_states)

        attn_output = attn_output.flatten(2)
        assert attn_output.is_contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None
    
module.Qwen2Attention = Qwen2NSA
trigger = None