import numpy as np
import torch
from torch import nn
from models.containers import Module
import sys
from models.DLCT.utils import PositionWiseFeedForward

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

        self.comment = comment

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class ScaledDotProductAttentionWithGoc(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttentionWithGoc, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        #TODO:v4-3新增参数
        self.new_fc_k = nn.Linear(d_model, h * d_k)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.comment = comment

        self.init_weights()
        # self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)
        #TODO：新增参数的初始化
        nn.init.xavier_uniform_(self.new_fc_k.weight)
        nn.init.constant_(self.new_fc_k.bias, 0)


    def func(self,a):
        b = a.min()
        if b < 0:
            c = a - b
        else:
            c = a
        return c
    def unfunc(self,a,addvalue):
        if addvalue > 0:
            c = a
        else:
            c = a + addvalue
        return c
    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None,aligns = None):
        '''
        此处的keys形状是(bs 99+50 512) values形状是(bs 99 512)
        '''
        encdoer_out = keys[:,:99,:]
        new_region = keys[:,99:,:]

        align = aligns
        b_s, nq = queries.shape[:2]
        nk = encdoer_out.shape[1]
        nv= values.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(encdoer_out).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nv, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        #TODO：step1,计算词向量与encoder_out的相似度.并且将相似度矩阵中的元素全部置换为正数。
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, 19, 99)
        if attention_weights is not None:
            att = att * attention_weights
        att = self.dropout(att)
        att = att.masked_fill(attention_mask, 0)
        att = self.func(att)
        att = att.masked_fill(attention_mask, 0)
        Attr = att[:, :, :, :50]
        Attg = att[:, :, :, 50:]
        #TODO：step2,计算词向量与new_region的相似度，并将相似度矩阵中的元素全部置换为正数。
        new_nk = new_region.shape[1]
        region_att_mask = attention_mask[:, :, :, :50]
        new_k = self.new_fc_k(new_region).view(b_s, new_nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        new_att = torch.matmul(q, new_k) / np.sqrt(self.d_k)
        new_att = self.dropout(new_att)
        new_att = new_att.masked_fill(region_att_mask, 0)
        new_att = self.func(new_att)
        new_att = new_att.masked_fill(region_att_mask, 0)
        # todo:第二步，形状：(bs h 19,50)*(bs 1 50 49) = (bs h 19 49)
        goatt = torch.matmul(new_att, align) / 25
        # todo: 第三步，求和（应用goc计算结果）
        Attg = Attg + goatt
        # todo:第四步，拼接还原
        New_att = torch.cat([Attr, Attg], -1)
        New_att = New_att.masked_fill(attention_mask, -np.inf)
        New_att = torch.softmax(New_att, -1)

        ##todo:v2.0
        # att = att.masked_fill(attention_mask, 0)
        # att = self.func(att)
        # att = att.masked_fill(attention_mask, 0)
        # Attr = att[:, :, :, :50]  # 词向量与区域特征计算出的注意力矩阵
        # Attg = att[:, :, :, 50:]  # 词向量与网格特征计算出的注意力矩阵
        # # todo:第二步，形状：(bs h 19,50)*(bs 1 50 49) = (bs h 19 49)
        # # sum_align= torch.sum(align,2).unsqueeze(2)
        # goatt = torch.matmul(Attr, align) / 25
        # # todo: 第三步，求和（应用goc计算结果）
        # Attg = Attg + goatt
        # # todo:第四步，拼接还原
        # New_att = torch.cat([Attr, Attg], -1)
        # New_att = New_att.masked_fill(attention_mask, -np.inf)
        # New_att = torch.softmax(New_att, -1)
        ##todo:v3.0
        # att = att.masked_fill(attention_mask, -np.inf)
        # Attr = att[:, :, :, :50]  # 词向量与区域特征计算出的注意力矩阵
        # Attg = att[:, :, :, 50:]  # 词向量与网格特征计算出的注意力矩阵
        # Attr = torch.softmax(Attr, -1)
        # Attg = torch.softmax(Attg, -1)
        # # todo:第二步，形状：(bs h 19,50)*(bs 1 50*49) = (bs h 19*49)
        # goatt = torch.matmul(att.masked_fill(attention_mask, 0)[:, :, :, :50], align)
        # goatt = torch.softmax(goatt, -1)
        # # todo: 第三步，求和（应用goc计算结果）
        # Attg = Attg + goatt
        # # todo:第四步，拼接还原
        # New_att = torch.cat([Attr, Attg], -1)
        # New_att = New_att.masked_fill(attention_mask, 0)

        out = torch.matmul(New_att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class ScaledDotProductWithBoxAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductWithBoxAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

        self.comment = comment

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, box_relation_embed_matrix, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights

        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        w_g = box_relation_embed_matrix
        w_a = att

        w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_a
        w_mn = torch.softmax(w_mn, -1)  ## bs * 8 * r * r

        att = self.dropout(w_mn)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class ScaledDotProductWithBoxExtrAtt(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None):

        super(ScaledDotProductWithBoxExtrAtt, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

        self.comment = comment

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, box_relation_embed_matrix,ExtrAtt, attention_mask=None, attention_weights=None):

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights

        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        w_g = box_relation_embed_matrix
        w_a = att

        w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_a
        w_mn = torch.softmax(w_mn, -1)  ## bs * 8 * r * r

        att = self.dropout(w_mn)
        # TODO：将原有的注意力矩阵和额外的注意力矩阵求和，随后通过softmax将矩阵中的元素归到0与1之间
        att = att+ExtrAtt
        att = torch.softmax(att, -1)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class MultiHeadBoxAttention(Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, comment=None):
        super(MultiHeadBoxAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductWithBoxAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values,box_relation_embed_matrix, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm,box_relation_embed_matrix, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values,box_relation_embed_matrix, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out


class MultiHeadAttention(Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, comment=None):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out


class MultiHeadAttentionWithGoc(Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, comment=None):
        super(MultiHeadAttentionWithGoc, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductAttentionWithGoc(d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None,aligns=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights,aligns)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, attention_mask, attention_weights,aligns)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out

class AttMatrixByScaledDotProductWithBox(nn.Module):
    '''
    将计算出的注意力矩阵作为返回值，而不做与V的矩阵乘法运算

    '''
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None):
        super(AttMatrixByScaledDotProductWithBox, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

        self.comment = comment

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)


    def forward(self, queries, keys, box_relation_embed_matrix, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights

        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        w_g = box_relation_embed_matrix
        w_a = att

        w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_a
        w_mn = torch.softmax(w_mn, -1)  ## bs * 8 * r * r

        att = self.dropout(w_mn)

        return att


class MultiHeadAttentionWithBoxExtrAtt(Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, comment=None):
        super(MultiHeadAttentionWithBoxExtrAtt, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductWithBoxExtrAtt(d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values,box_relation_embed_matrix, ExtrAtt,attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm,box_relation_embed_matrix, ExtrAtt,attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values,box_relation_embed_matrix, ExtrAtt,attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out
