import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from models.DLCT.attention import MultiHeadAttention
from models.DLCT.attention import MultiHeadAttentionWithGoc
from models.DLCT.utils import sinusoid_encoding_table, PositionWiseFeedForward
from models.containers import Module, ModuleList
from models.DLCT.DGOC import VisualCrossAtt

class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)# mask self attention
        self.enc_att = MultiHeadAttentionWithGoc(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)#cross attention

        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att, pos,aligns):
        # MHA+AddNorm
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        self_att = self_att * mask_pad # 过滤pad位置处的信息
        # MHA+AddNorm

        #TODO：v4-2新代码
        region_pos =  pos[:,:50,:]
        new_pos = torch.cat([pos,region_pos],1)
        key = enc_output + new_pos

        enc_att = self.enc_att(self_att, key, enc_output[:,:99,:], mask_enc_att,attention_weights = None,aligns = aligns) #enc_att.shape = bs 19 512
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))
        enc_att = enc_att * mask_pad# 过滤pad位置处的信息
        # FFN+AddNorm
        ff = self.pwff(enc_att)
        ff = ff * mask_pad# 过滤pad位置处的信息
        return ff


class TransformerDecoderLayer(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
                          enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                          enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec
        #TODO：声明visual_Cross_att
        self.visual_CA = VisualCrossAtt(d_model, d_k, d_v, h, d_ff,dropout,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder, pos,aligns):
        # input (b_s, seq_len)
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)# 上三角矩阵，即仅主对角线及以上为1，其余为0
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat(
                [self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)

        if encoder_output.shape[0] != pos.shape[0]:
            assert encoder_output.shape[0] % pos.shape[0] == 0
            beam_size = int(encoder_output.shape[0] / pos.shape[0])
            shape = (pos.shape[0], beam_size, pos.shape[1], pos.shape[2])
            pos = pos.unsqueeze(1)  # bs * 1 * 50 * 512
            pos = pos.expand(shape)  # bs * 5 * 50 * 512
            pos = pos.contiguous().flatten(0, 1)  # (bs*5) * 50 * 512

        #TODO:v4-2在新增代码
        new_region = self.visual_CA(encoder_output,mask_encoder,pos)
        new_visual_feauture = torch.cat([encoder_output,new_region],dim=1) #(bs,99+50,512) #第二个维度的构成：region & grid & new_region

        for i, l in enumerate(self.layers):
            # print(out.shape)
            # print(encoder_output.shape)
            # print(mask_queries.shape)
            # print(mask_self_attention.shape)
            # print(mask_encoder.shape)
            # print(pos.shape)
            out = l(out, new_visual_feauture, mask_queries, mask_self_attention, mask_encoder, pos=pos,aligns = aligns)
            # print('decoder layer out')
            # print(out[11])

        out = self.fc(out)
        # print('decoder out')
        # print(out[11])
        return F.log_softmax(out, dim=-1)
