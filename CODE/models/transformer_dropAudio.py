import torch
from torch import Tensor, Union, Callable, Optional
from torch.nn import Module, MultiheadAttention, Linear, Dropout, LayerNorm, ModuleList
import torch.nn.functional as F
import copy

class TransformerDecoder(Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, 
                memory_comments: Tensor, memory_video: Tensor, 
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        output = tgt

        for mod in self.layers:
            output = mod(output, memory_comments, memory_video, tgt_mask=tgt_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn_comments = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        #self.multihead_attn_audio = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
        #                                         **factory_kwargs)
        self.multihead_attn_video = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
        self.dropout_ff = Dropout(dropout)
        self.norm_ff = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.norm_first = norm_first

        self.norm_sa = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout_sa = Dropout(dropout)

        self.norm_mha_comments = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout_mha_comments = Dropout(dropout)

        #self.norm_mha_audio = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        #self.dropout_mha_audio = Dropout(dropout)

        self.norm_mha_video = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout_mha_video = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory_comments: Tensor,
        #memory_audio: Tensor,
        memory_video: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
    ) -> Tensor:
       
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm_sa(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._mha_block_comments(self.norm_mha_comments(x), memory_comments, None, None)
        #    x = x + self._mha_block_audio(self.norm_mha_audio(x), memory_audio, None, None)
            x = x + self._mha_block_video(self.norm_mha_video(x), memory_video, None, None)
            x = x + self._ff_block(self.norm_ff(x))
        else:
            x = self.norm_sa(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm_mha_comments(x + self._mha_block_comments(x, memory_comments, None, None)) # cross attention from target to context comm
        #    x = self.norm_mha_audio(x + self._mha_block_audio(x, memory_audio, None, None)) # cross attention from context comm to audio
            x = self.norm_mha_video(x + self._mha_block_video(x, memory_video, None, None)) # cross attention from audio to video
            x = self.norm_ff(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout_sa(x)

    # multihead attention blocks
    def _mha_block_comments(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
       
        x = self.multihead_attn_comments(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout_mha_comments(x)
    
    """  def _mha_block_audio(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.multihead_attn_audio(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout_mha_audio(x) 
    """
    
    def _mha_block_video(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        
        x = self.multihead_attn_video(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout_mha_video(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout_ff(x)


class VCDecoderLayer(Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn_comments = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        self.multihead_attn_video = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
        self.dropout_ff = Dropout(dropout)
        self.norm_ff = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.norm_first = norm_first

        self.norm_sa = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout_sa = Dropout(dropout)

        self.norm_mha_comments = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout_mha_comments = Dropout(dropout)

        self.norm_mha_video = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout_mha_video = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory_comments: Tensor,
        memory_audio: Tensor,
        memory_video: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
    ) -> Tensor:
        x = tgt
        
        if self.norm_first:
            x = x + self._sa_block(self.norm_sa(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._mha_block_comments(self.norm_mha_comments(x), memory_comments, None, None)
            x = x + self._mha_block_video(self.norm_mha_video(x), memory_video, None, None)
            x = x + self._ff_block(self.norm_ff(x))
        else:
            x = self.norm_sa(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm_mha_comments(x + self._mha_block_comments(x, memory_comments, None, None))
            x = self.norm_mha_video(x + self._mha_block_video(x, memory_video, None, None))
            x = self.norm_ff(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout_sa(x)

    # multihead attention blocks
    def _mha_block_comments(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.multihead_attn_comments(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout_mha_comments(x)
    
    def _mha_block_video(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.multihead_attn_video(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout_mha_video(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout_ff(x)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

