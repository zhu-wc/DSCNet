from torch.nn import functional as F
from models.DLCT.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.DLCT.attention import MultiHeadBoxAttention as MultiHeadAttention

from ..relative_embedding import BoxRelationalEmbedding, GridRelationalEmbedding, AllRelationalEmbedding
from models.DLCT.attention import AttMatrixByScaledDotProductWithBox
from models.DLCT.attention import MultiHeadAttentionWithBoxExtrAtt
from models.DLCT.EGOC import MultiHeadAttentionGLU
import numpy as np

class SelfAtt(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(SelfAtt, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights, attention_mask=None, attention_weights=None,
                pos=None):
        q = queries + pos
        k = keys + pos
        att = self.mhatt(q, k, values, relative_geometry_weights, attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff

class SelfAttWithBoxExtrAtt(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(SelfAttWithBoxExtrAtt, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttentionWithBoxExtrAtt(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights,ExtrAtt, attention_mask=None, attention_weights=None,
                pos=None):
        q = queries + pos
        k = keys + pos
        att = self.mhatt(q, k, values, relative_geometry_weights, ExtrAtt,attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff

class GOC(nn.Module): #针对网格特征的对象注意力运算
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(GOC, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttentionGLU(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.AttMatrix = AttMatrixByScaledDotProductWithBox(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)
    def forward(self, queries, keys, values, region2grid,region2region,align,attention_mask_region,attention_mask_grid,attention_weights=None,
                pos_query=None, pos_key=None):
        '''
        q的是区域，网格作为k、v ; pos_query是区域的绝对位置编码 ; pos_key是网格的绝对位置编码
        region2grid应该是50*49的矩阵，其表示区域到网格的相对位置注意力矩阵，这一点与LCCA中构建的任何一个矩阵都不同，需要自己重新构建
        （DLCT中关于相对位置注意力矩阵只有49*49，50*50，50*99，49*99）
        align是区域到网格的对齐矩阵，其形状为50*49
        '''
        align_T = align.permute(0,2,1)
        # TODO:第一步，cross-attention
        q = queries + pos_query
        k = keys + pos_key
        att = self.mhatt(q, k, values, region2grid, attention_mask_grid, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att) #此处的ff可以看作是第一步cross-att的运算结果
        # TODO：第二步，scaled dot-product attention
        object_att = self.AttMatrix(ff,ff,region2region,attention_mask_region,attention_weights)  #50*50的矩阵
        # TODO: 第三步，OAM映射 50*50->49*49
        align = align.unsqueeze(1)
        align_T = align_T.unsqueeze(1)
        #（50，50)*(50,49)=(50,49)，其中(50,49)是对齐矩阵
        L = torch.matmul(object_att,align)
        # (49*50)*(50*49)=(49,49)，其中(49,50)是对其矩阵，是上一行使用的对齐矩阵的转置
        out = torch.matmul(align_T,L)
        out = torch.softmax(out,-1)
        return out#49*49的矩阵


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.SA_wBox = nn.ModuleList([SelfAtt(d_model, d_k, d_v, h, d_ff, dropout,
                                                         identity_map_reordering=identity_map_reordering,
                                                         attention_module=attention_module,
                                                         attention_module_kwargs=attention_module_kwargs)
                                            for _ in range(N)])
        self.layers_grid = nn.ModuleList([SelfAttWithBoxExtrAtt(d_model, d_k, d_v, h, d_ff, dropout,
                                                       identity_map_reordering=identity_map_reordering,
                                                       attention_module=attention_module,
                                                       attention_module_kwargs=attention_module_kwargs)
                                          for _ in range(N)])

        self.layers_GOC = nn.ModuleList([GOC(d_model, d_k, d_v, h, d_ff, dropout,
                                                       identity_map_reordering=identity_map_reordering,
                                                       attention_module=attention_module,
                                                       attention_module_kwargs=attention_module_kwargs)
                                          for _ in range(N)])
        
        self.padding_idx = padding_idx

        self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

    def forward(self, regions, grids, boxes, aligns, attention_weights=None, region_embed=None, grid_embed=None):
        # regions.shape =  (b_s, 50, 512)
        # grids.shape = (b_s,49,512)

        attention_mask_region = (torch.sum(regions == 0, -1) != 0).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        attention_mask_grid = (torch.sum(grids == 0, -1) != 0).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        # box embedding
        relative_geometry_embeddings = AllRelationalEmbedding(boxes)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
                                              self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        n_regions = regions.shape[1]  # 50
        n_grids = grids.shape[1]
        region2region = relative_geometry_weights[:, :, :n_regions, :n_regions]  #50*50
        grid2grid = relative_geometry_weights[:, :, n_regions:, n_regions:]   #49*49
        #TODO: 针对新提出的两个模块，构建两个新的相对位置矩阵
        region2grid = relative_geometry_weights[:, :, :n_regions , n_regions:]  #50*49
        grid2region = relative_geometry_weights[:, :, n_regions: , :n_regions]  #49*50

        out_region = regions
        out_grid = grids
        #TODO: 考虑到Align矩阵具有三个维度，所以在进行转置时需要从三个维度上考虑
        align_region2grid = aligns
        align_grid2region = aligns.permute(0,2,1)

        for l_sab, l_grid ,l_goc in zip(self.SA_wBox, self.layers_grid,self.layers_GOC):
            # RRAtt = l_rrc(out_grid,out_region,out_region,grid2region,grid2grid,align_grid2region,attention_mask_region,attention_mask_grid,attention_weights,
            #                                                     pos_query=grid_embed,pos_key=region_embed)

            GOAtt = l_goc(out_region,out_grid,out_grid,region2grid,region2region,align_region2grid,attention_mask_region,attention_mask_grid,attention_weights,
                                                                 pos_query=region_embed , pos_key=grid_embed)


            # out_region = l_region(out_region, out_region, out_region, region2region,RRAtt, attention_mask_region,
            #                       attention_weights, pos=region_embed)
            out_region = l_sab(out_region, out_region, out_region, region2region, attention_mask_region,
                                  attention_weights, pos=region_embed)

            out_grid = l_grid(out_grid, out_grid, out_grid, grid2grid,GOAtt, attention_mask_grid, attention_weights,
                              pos=grid_embed)

        out = torch.cat([out_region, out_grid], dim=1)
        attention_mask = torch.cat([attention_mask_region, attention_mask_grid], dim=-1)
        return out, attention_mask


class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc_region = nn.Linear(d_in, self.d_model)
        self.dropout_region = nn.Dropout(p=self.dropout)
        self.layer_norm_region = nn.LayerNorm(self.d_model)

        self.fc_grid = nn.Linear(d_in, self.d_model)
        self.dropout_grid = nn.Dropout(p=self.dropout)
        self.layer_nrom_grid = nn.LayerNorm(self.d_model)

    def forward(self, regions, grids, boxes, aligns, attention_weights=None, region_embed=None, grid_embed=None):
        mask_regions = (torch.sum(regions, dim=-1) == 0).unsqueeze(-1)
        mask_grids = (torch.sum(grids, dim=-1) == 0).unsqueeze(-1)
        # print('\ninput', input.view(-1)[0].item())
        out_region = F.relu(self.fc_region(regions))
        out_region = self.dropout_region(out_region)
        out_region = self.layer_norm_region(out_region)
        out_region = out_region.masked_fill(mask_regions, 0)

        out_grid = F.relu(self.fc_grid(grids))
        out_grid = self.dropout_grid(out_grid)
        out_grid = self.layer_nrom_grid(out_grid)
        out_grid = out_grid.masked_fill(mask_grids, 0)

        # print('out4',out[11])
        return super(TransformerEncoder, self).forward(out_region, out_grid, boxes, aligns,
                                                       attention_weights=attention_weights,
                                                       region_embed=region_embed, grid_embed=grid_embed)
