## HINT
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from sklearn.cluster import SpectralClustering
import warnings

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
class Inter_CacheModulation(nn.Module):
    def __init__(self, in_c=3):
        super(Inter_CacheModulation, self).__init__()

        self.align = nn.AdaptiveAvgPool2d(in_c)
        self.conv_width = nn.Conv1d(in_channels=in_c, out_channels=2*in_c, kernel_size=1)
        self.gatingConv = nn.Conv1d(in_channels=in_c, out_channels=in_c, kernel_size=1)

    def forward(self, x1,x2):
        C = x1.shape[-1]
        x2_pW = self.conv_width(self.align(x2)+x1)
        scale,shift = x2_pW.chunk(2, dim=1)
        x1_p = x1*scale+shift
        x1_p = x1_p * F.gelu(self.gatingConv(x1_p))
        return x1_p


class Intra_CacheModulation(nn.Module):
    def __init__(self,embed_dim=48):
        super(Intra_CacheModulation, self).__init__()

        self.down = nn.Conv1d(embed_dim, embed_dim//2, kernel_size=1)
        self.up = nn.Conv1d(embed_dim//2, embed_dim, kernel_size=1)
        self.gatingConv = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1)


    def forward(self, x1,x2):
        x_gated = F.gelu(self.gatingConv(x2+x1)) * (x2+x1)
        x_p = self.up(self.down(x_gated))  
        return x_p

class ReGroup(nn.Module):
    def __init__(self, groups=[1,1,2,4]):
        super(ReGroup, self).__init__()
        self.gourps = groups

    def forward(self, query,key,value):
        C = query.shape[1]
        channel_features = query.mean(dim=0)
        correlation_matrix = torch.corrcoef(channel_features)

        mean_similarity = correlation_matrix.mean(dim=1)
        _, sorted_indices = torch.sort(mean_similarity, descending=True) 

        query_sorted = query[:, sorted_indices, :]
        key_sorted = key[:, sorted_indices, :]
        value_sorted = value[:, sorted_indices, :]

        query_groups = []
        key_groups = []
        value_groups = []
        start_idx = 0
        total_ratio = sum(self.gourps)
        group_sizes = [int(ratio / total_ratio * C) for ratio in self.gourps]

        for group_size in group_sizes:
            end_idx = start_idx + group_size
            query_groups.append(query_sorted[:, start_idx:end_idx, :])  
            key_groups.append(key_sorted[:, start_idx:end_idx, :])  
            value_groups.append(value_sorted[:, start_idx:end_idx, :])  
            start_idx = end_idx

        return query_groups,key_groups,value_groups


# def CalculateCurrentLayerCache(x,dim=128,groups=[1,1,2,4]):
#     lens = len(groups)
#     ceil_dim = dim #* max_value // sum_value 
#     for i in range(lens):
#         qv_cache_f = x[i].clone().detach()
#         qv_cache_f=torch.mean(qv_cache_f,dim=0,keepdim=True).detach()
#         update_elements = F.interpolate(qv_cache_f.unsqueeze(1), size=(ceil_dim, ceil_dim), mode='bilinear', align_corners=False)
#         c_i = qv_cache_f.shape[-1]
                
#         if i==0:
#             qv_cache = update_elements * c_i // dim
#         else:
#             qv_cache = qv_cache + update_elements * c_i // dim
                
#     return qv_cache.squeeze(1)


def CalculateCurrentLayerCache(x, dim=128, groups=[1,1,2,4]):
    lens = len(groups)
    ceil_dim = dim

    qv_cache = None
    for i in range(lens):
        qv_cache_f = x[i].detach()
        qv_cache_f = torch.mean(qv_cache_f, dim=0, keepdim=True)  # [1, ci, ci]

        update_elements = F.interpolate(
            qv_cache_f.unsqueeze(1), size=(ceil_dim, ceil_dim),
            mode='bilinear', align_corners=False
        )  # [1, 1, dim, dim]

        c_i = qv_cache_f.shape[-1]
        scale = float(c_i) / float(dim)   # 用浮点比例，不要 //

        if qv_cache is None:
            qv_cache = update_elements * scale
        else:
            qv_cache = qv_cache + update_elements * scale

    return qv_cache.squeeze(1)  # [1, dim, dim]


class SpatialWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, bias=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.ws = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 空间分支投影：独立一套更稳（也可以复用主 qkv，但独立一般更好）
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def _pad_to_window(self, x):
        # x: [b,c,h,w]
        b, c, h, w = x.shape
        pad_h = (self.ws - h % self.ws) % self.ws
        pad_w = (self.ws - w % self.ws) % self.ws
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (left,right,top,bottom) for last two dims
        return x, pad_h, pad_w

    def forward(self, x):
        # x: [b,c,h,w]
        b, c, h, w = x.shape
        x_pad, pad_h, pad_w = self._pad_to_window(x)
        _, _, hp, wp = x_pad.shape
        nh = hp // self.ws
        nw = wp // self.ws

        qkv = self.qkv_dw(self.qkv(x_pad))
        q_map, k_map, v_map = qkv.chunk(3, dim=1)  # [b,c,hp,wp]

        # 切 window，并做多头： (b*nh*nw, head, ws*ws, d)
        q = rearrange(q_map, 'b (head d) (nh ws1) (nw ws2) -> (b nh nw) head (ws1 ws2) d',
                      head=self.num_heads, nh=nh, nw=nw, ws1=self.ws, ws2=self.ws)
        k = rearrange(k_map, 'b (head d) (nh ws1) (nw ws2) -> (b nh nw) head (ws1 ws2) d',
                      head=self.num_heads, nh=nh, nw=nw, ws1=self.ws, ws2=self.ws)
        v = rearrange(v_map, 'b (head d) (nh ws1) (nw ws2) -> (b nh nw) head (ws1 ws2) d',
                      head=self.num_heads, nh=nh, nw=nw, ws1=self.ws, ws2=self.ws)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [(b*nh*nw), head, N, N]
        attn = attn.softmax(dim=-1)
        out = attn @ v                                  # [(b*nh*nw), head, N, d]

        # 拼回 feature map
        out = rearrange(out, '(b nh nw) head (ws1 ws2) d -> b (head d) (nh ws1) (nw ws2)',
                        b=b, nh=nh, nw=nw, head=self.num_heads, ws1=self.ws, ws2=self.ws)

        out = out[:, :, :h, :w]  # 去 padding
        out = self.proj(out)
        return out


# class Attention(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(4, 1, 1))

#         self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
#         self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.group =[1,2,2,3] 

#         self.intra_modulator = Intra_CacheModulation(embed_dim=dim)

#         self.inter_modulator1 = Inter_CacheModulation(in_c=1*dim//8)
#         self.inter_modulator2 = Inter_CacheModulation(in_c=2*dim//8)
#         self.inter_modulator3 = Inter_CacheModulation(in_c=2*dim//8)
#         self.inter_modulator4 = Inter_CacheModulation(in_c=3*dim//8)
#         self.inter_modulators = [self.inter_modulator1,self.inter_modulator2,self.inter_modulator3,self.inter_modulator4]

#         self.regroup = ReGroup(self.group)
#         self.dim=dim

#     def forward(self, x ,qv_cache=None):
#         b,c,h,w = x.shape

#         # b = x.size(0)
#         # c = int(x.size(1))   # 关键：强制变成 Python int
#         # h = x.size(2)
#         # w = x.size(3)

#         qkv = self.qkv_dwconv(self.qkv(x))
#         q,k,v = qkv.chunk(3, dim=1)   
    
#         q = rearrange(q, 'b c h w -> b c (h w)')
#         k = rearrange(k, 'b c h w -> b c (h w)')
#         v = rearrange(v, 'b c h w -> b c (h w)')

#         qu,ke,va = self.regroup(q,k,v)
#         attScore = []
#         tmp_cache=[]
#         for index in range(len(self.group)):

#             query_head = qu[index]
#             key_head   = ke[index]

#             query_head = torch.nn.functional.normalize(query_head, dim=-1)
#             key_head = torch.nn.functional.normalize(key_head, dim=-1)

#             attn = (query_head @ key_head.transpose(-2, -1)) * self.temperature[index,:,:]
#             attn = attn.softmax(dim=-1)

#             attScore.append(attn)#CxC
#             t_cache = query_head.clone().detach()+key_head.clone().detach()
#             tmp_cache.append(t_cache)
        
#         tmp_caches = torch.cat(tmp_cache, 1)
#         # Inter Modulation
#         out=[]


#         if qv_cache is not None:
#             if qv_cache.shape[-1]!=c:
                
#                 qv_cache = F.adaptive_avg_pool2d(qv_cache,c)


#         # if qv_cache is not None:
#         #     # 可选：保证 device/dtype 一致
#         #     qv_cache = qv_cache.to(device=x.device, dtype=x.dtype)

#         #     if qv_cache.shape[-1] != c:
#         #         # 关键：output_size 用 (c, c)，且 c 是 Python int
#         #         qv_cache = F.adaptive_avg_pool2d(qv_cache, output_size=(c, c))



#         for i in range(4):
#             if qv_cache is not None:
#                 inter_modulator = self.inter_modulators[i]
#                 attScore[i] = inter_modulator(attScore[i],qv_cache)+attScore[i]
#                 out.append(attScore[i] @ va[i])
#             else:
#                 out.append(attScore[i] @ va[i])
                
#         update_factor=0.9
#         if qv_cache is not None:
            
#             update_elements = CalculateCurrentLayerCache(attScore,c,self.group)
#             qv_cache = qv_cache*update_factor + update_elements*(1-update_factor)
#         else:
#             qv_cache = CalculateCurrentLayerCache(attScore,c,self.group)
#             qv_cache = qv_cache*update_factor

#         out_all = torch.concat(out, 1)
#         # Intra Modulation
#         out_all = self.intra_modulator(out_all,tmp_caches)+out_all

#         out_all = rearrange(out_all, 'b  c (h w) -> b c h w', h=h, w=w)
#         out_all = self.project_out(out_all)
#         return [out_all,qv_cache]


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        # ===== 原有通道分支（保持不动核心逻辑）=====
        self.temperature = nn.Parameter(torch.ones(4, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.group =[1,2,2,3]
        self.regroup = ReGroup(self.group)
        self.dim = dim

        self.intra_modulator = Intra_CacheModulation(embed_dim=dim)

        self.inter_modulator1 = Inter_CacheModulation(in_c=1*dim//8)
        self.inter_modulator2 = Inter_CacheModulation(in_c=2*dim//8)
        self.inter_modulator3 = Inter_CacheModulation(in_c=2*dim//8)
        self.inter_modulator4 = Inter_CacheModulation(in_c=3*dim//8)
        self.inter_modulators = [self.inter_modulator1, self.inter_modulator2, self.inter_modulator3, self.inter_modulator4]

        # ===== 新增：空间分支（window spatial attn）=====
        self.spatial_attn = SpatialWindowAttention(dim=dim, num_heads=num_heads, window_size=8, bias=bias)

        # 融合系数：初始化为 0，让网络从原版起步，训练更稳
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, qv_cache=None):
        b, c, h, w = x.shape

        # ---------- 空间分支（直接从 x 走）----------
        out_spatial = self.spatial_attn(x)                         # [b,c,h,w]
        out_spatial_t = rearrange(out_spatial, 'b c h w -> b c (h w)')  # [b,c,hw]

        # ---------- 通道分支（你的原逻辑）----------
        qkv = self.qkv_dwconv(self.qkv(x))
        q_map, k_map, v_map = qkv.chunk(3, dim=1)                  # [b,c,h,w]

        q = rearrange(q_map, 'b c h w -> b c (h w)')
        k = rearrange(k_map, 'b c h w -> b c (h w)')
        v = rearrange(v_map, 'b c h w -> b c (h w)')

        qu, ke, va = self.regroup(q, k, v)

        attScore = []
        tmp_cache = []
        for index in range(len(self.group)):
            query_head = F.normalize(qu[index], dim=-1)
            key_head   = F.normalize(ke[index], dim=-1)

            attn = (query_head @ key_head.transpose(-2, -1)) * self.temperature[index, :, :]
            attn = attn.softmax(dim=-1)

            attScore.append(attn)
            t_cache = query_head.detach() + key_head.detach()
            tmp_cache.append(t_cache)

        tmp_caches = torch.cat(tmp_cache, 1)  # [b,c,hw]

        # cache 尺寸对齐（保持你现在的写法）
        if qv_cache is not None:
            if qv_cache.shape[-1] != c:
                qv_cache = F.adaptive_avg_pool2d(qv_cache, c)

        out = []
        for i in range(4):
            if qv_cache is not None:
                attScore[i] = self.inter_modulators[i](attScore[i], qv_cache) + attScore[i]
            out.append(attScore[i] @ va[i])   # [b,ci,hw]

        # 更新 cache（保持你的 EMA）
        update_factor = 0.9
        if qv_cache is not None:
            update_elements = CalculateCurrentLayerCache(attScore, c, self.group)
            qv_cache = qv_cache * update_factor + update_elements * (1 - update_factor)
        else:
            qv_cache = CalculateCurrentLayerCache(attScore, c, self.group)
            qv_cache = qv_cache * update_factor

        out_channel_t = torch.cat(out, 1)     # [b,c,hw]

        # ---------- 融合：通道 + 空间 ----------
        out_all = out_channel_t + self.alpha * out_spatial_t

        # Intra modulation（你原来对 out_all + tmp_caches 的融合）
        out_all = self.intra_modulator(out_all, tmp_caches) + out_all

        # 回到 map
        out_all = rearrange(out_all, 'b c (h w) -> b c h w', h=h, w=w)
        out_all = self.project_out(out_all)

        return [out_all, qv_cache]



class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, isAtt):
        super(TransformerBlock, self).__init__()
        self.isAtt = isAtt
        if self.isAtt:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self,inputs):
        x = inputs[0]
        qv_cache = inputs[1]
        if self.isAtt:
            x_tmp = x
            [x_att,qv_cache] = self.attn(self.norm1(x),qv_cache=qv_cache)
            x = x_tmp + x_att
        x = x + self.ffn(self.norm2(x))

        return [x,qv_cache]




class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x




class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)




class HINT1(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 32,
        num_blocks = [2,4,4,6], 
        num_refinement_blocks = 2,
        heads = [8,8,8,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
        dual_pixel_task = False,
        qv_cache=None
    ):

        super(HINT1, self).__init__()

        self.qv_cache=qv_cache

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.to_img_level3 = nn.Conv2d(int(dim*2**2), out_channels, 3, 1, 1, bias=bias)
        self.to_img_level2 = nn.Conv2d(int(dim*2**1), out_channels, 3, 1, 1, bias=bias)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=False) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=False) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1))
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=False) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=True) for i in range(num_blocks[1])])
        
        self.up4_3 = Upsample(int(dim*2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=True) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=True) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=True) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=True) for i in range(num_refinement_blocks)])
        

        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)

            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        
        out_enc_level1,self.qv_cache = self.encoder_level1([inp_enc_level1,self.qv_cache])
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2,self.qv_cache = self.encoder_level2([inp_enc_level2,self.qv_cache])

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3,self.qv_cache = self.encoder_level3([inp_enc_level3,self.qv_cache]) 

        inp_enc_level4 = self.down3_4(out_enc_level3) 
        latent,self.qv_cache = self.latent([inp_enc_level4,self.qv_cache])

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3, self.qv_cache = self.decoder_level3([inp_dec_level3, self.qv_cache])

        # [新增] 输出 Level 3 的预测图 (需要上采样到原图尺寸)
        # 【新增】计算 Level 3 的中间输出
        # ============================================================
        img_level3 = None
        if self.training: # 只有训练时才计算，节省推理时间
            img_level3 = self.to_img_level3(out_dec_level3)
            # 上采样到与输入图像一致的大小 (B, 3, H, W)
            img_level3 = F.interpolate(img_level3, size=inp_img.shape[-2:], mode='bilinear', align_corners=False)
            # 如果是 Residual Learning，通常中间层也加上输入会更好收敛，或者直接让中间层拟合 GT
            if not self.dual_pixel_task:
                 img_level3 = img_level3 + inp_img

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2,self.qv_cache = self.decoder_level2([inp_dec_level2,self.qv_cache]) 

        # ============================================================
        # 【新增】计算 Level 2 的中间输出
        # ============================================================
        img_level2 = None
        if self.training:
            img_level2 = self.to_img_level2(out_dec_level2)
            img_level2 = F.interpolate(img_level2, size=inp_img.shape[-2:], mode='bilinear', align_corners=False)
            if not self.dual_pixel_task:
                 img_level2 = img_level2 + inp_img

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1,self.qv_cache = self.decoder_level1([inp_dec_level1,self.qv_cache])

        
        out_dec_level1,self.qv_cache = self.refinement([out_dec_level1,self.qv_cache])

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_final = self.output(out_dec_level1)
        ###########################
        else:
            out_final = self.output(out_dec_level1) + inp_img

        # ============================================================
        # 【新增】返回值逻辑
        # ============================================================
        if self.training:
            # 返回列表：[最终输出, Level2输出, Level3输出]
            return [out_final, img_level2, img_level3]
        else:
            # 验证/测试时只返回最终结果
            return out_final








import torch
from thop import profile
import sys
import os

# 初始化模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义模型
model = HINT1().to(device)

# 假设输入图像的尺寸为3x256x256
input_tensor = torch.randn(1, 3, 256, 256).to(device)


# 计算FLOPS 和 Params
macs, params = profile(model, inputs=(input_tensor,))
print(f"model-Flops:{macs/1e9:.2f}G, Params:{params/1e6:2f}M")