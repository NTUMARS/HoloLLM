# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math
import functools
import copy

import torch
from torch import nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)
from ..components import RMSNorm
from flash_attn import flash_attn_func

import open_clip
from torchvision import models
from torchvision.transforms import Resize
default_linear_init = nn.init.xavier_uniform_


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim -
             1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.embedding = self._create_positional_embeddings()

    def _create_positional_embeddings(self):
        position = torch.arange(0, self.max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2).float() * (-math.log(10000.0) / self.dim)
        )
        embeddings = torch.zeros(self.max_seq_len, self.dim)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
        return embeddings.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        return self.embedding[:, :x.size(1), :].to(x.device)
    
# Input transformation network (T-net)
class TNet(nn.Module):
    def __init__(self, input_dim=3):
        super(TNet, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, input_dim * input_dim)
        self.fc3.weight.data.fill_(0)
        self.fc3.bias.data.fill_(0)
        self.input_dim = input_dim
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=False)[0]  # Max pooling
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.input_dim, self.input_dim)
        return x

# PointNet feature extraction network
class PointNetEncoder(nn.Module):
    def __init__(self, input_dim=3):
        super(PointNetEncoder, self).__init__()
        self.tnet = TNet(input_dim=input_dim)
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
    def forward(self, x):
        # Apply input transformation
        trans = self.tnet(x)
        x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1)
        
        # Feature extraction layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Max pooling
        # x = torch.max(x, 2, keepdim=False)[0]
        return x

# PointNet classification network
class PointNet(nn.Module):
    def __init__(self, input_dim=3):
        super(PointNet, self).__init__()
        self.encoder = PointNetEncoder(input_dim=input_dim)
        # self.fc1 = nn.Linear(1024, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, num_classes)
        self.initialize_parameters()
    def forward(self, x):
        # Feature extraction
        x = self.encoder(x)
        
        # Fully connected layers for classification
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x
    
    def initialize_parameters(self):
        # Initialize convolution layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            # Initialize fully connected layers
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            # Initialize the transformation matrix (TNet) to be the identity matrix
            elif isinstance(m, nn.Linear) and m.out_features == 9:  # for TNet output transformation matrix
                nn.init.constant_(m.weight, 0)
                nn.init.eye_(m.weight.data)
                nn.init.constant_(m.bias, 0)

# for wifi
class WiFi_Encoder(nn.Module):
    """Our RoadSeg takes rgb and another (depth or normal) as input,
    and outputs freespace predictions.
    """
    def __init__(self):
        super(WiFi_Encoder, self).__init__()
        resnet_raw_model1 = models.resnet18(pretrained=True)
        self.encoder_conv1_p1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.encoder_bn1_p1 = resnet_raw_model1.bn1
        self.encoder_relu_p1 = resnet_raw_model1.relu
        self.encoder_maxpool_p1 = resnet_raw_model1.maxpool
        self.encoder_layer1_p1 = resnet_raw_model1.layer1
        self.encoder_layer2_p1 = resnet_raw_model1.layer2
        self.encoder_layer3_p1 = resnet_raw_model1.layer3
        self.encoder_layer4_p1 = resnet_raw_model1.layer4

    def forward(self,x): #16,2,3,114,32
        # x = x.float()

        x = torch.transpose(x, 2, 3) #16,2,114,3,32
        x = torch.flatten(x, 3, 4)# 16,2,114,96
        # torch_resize = Resize([136,32])
        # x = torch_resize(x) #136,32

        x = x.unsqueeze(dim=2)

        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)

        x = self.encoder_conv1_p1(x)  ##16,2,136,136
        x = self.encoder_bn1_p1(x)  ##size16,64,136,136
        x = self.encoder_relu_p1(x)  ##size(1,64,192,624)

        x = self.encoder_layer1_p1(x)
        x = self.encoder_layer2_p1(x)
        x = self.encoder_layer3_p1(x)
        x = self.encoder_layer4_p1(x)

        x = x.view(B, T, x.shape[1], x.shape[2], x.shape[3])
        # x = x.mean(dim=1)
        return x

# Define the basic block for 1D ResNet
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

# Define the 1D ResNet-18 model
class RFID_Encoder(nn.Module):
    def __init__(self, num_classes=55):
        super(RFID_Encoder, self).__init__()

        self.conv1 = nn.Conv1d(23, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(BasicBlock1D(in_channels, out_channels, stride, downsample))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            x = x.float()
        else:
            x = x.half()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)  # [batchsize, 512]

        return x


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=default_linear_init,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=default_linear_init,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=default_linear_init,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=default_linear_init,
        )

        self.flash = True
        self.k_cache, self.v_cache = None, None

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], prompt=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.k_cache is None or self.v_cache is None:
            keys, values = xk, xv
        else:
            self.k_cache = self.k_cache.to(xk)
            self.v_cache = self.v_cache.to(xv)
            self.k_cache[:bsz, start_pos: start_pos + seqlen, :, :] = xk
            self.v_cache[:bsz, start_pos: start_pos + seqlen, :, :] = xv
            keys = self.k_cache[:bsz, :start_pos + seqlen]
            values = self.v_cache[:bsz, :start_pos + seqlen]

        output = flash_attn_func(
            xq, keys, values, dropout_p=0.0, causal=mask is not None)
        output = output.contiguous().view(bsz, seqlen, -1)

        return self.wo(output)

    def allocate_kv_cache(self, max_batch_size: int, max_seq_len: int) -> None:
        kv_cache_shape = (max_batch_size, max_seq_len,
                          self.n_local_heads, self.head_dim)
        if self.k_cache is None or self.k_cache.size() != kv_cache_shape:
            self.k_cache = torch.empty(kv_cache_shape)
        if self.v_cache is None or self.v_cache.size() != kv_cache_shape:
            self.v_cache = torch.empty(kv_cache_shape)

    def destroy_kv_cache(self) -> None:
        self.k_cache, self.v_cache = None, None


class CrossAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=default_linear_init,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=default_linear_init,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=default_linear_init,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=default_linear_init,
        )

        self.flash = True
        self.k_cache, self.v_cache = None, None

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], prompt=None):
        # the x_q would be the learnable queries
        bsz, seqlen_q, _ = x_q.shape
        _, seqlen_kv, _ = x_kv.shape
        xq, xk, xv = self.wq(x_q), self.wk(x_kv), self.wv(x_kv)

        xq = xq.view(bsz, seqlen_q, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen_kv, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen_kv, self.n_local_heads, self.head_dim)

        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis) 

        if self.k_cache is None or self.v_cache is None:
            keys, values = xk, xv
        else:
            self.k_cache = self.k_cache.to(xk)
            self.v_cache = self.v_cache.to(xv)
            self.k_cache[:bsz, start_pos: start_pos + seqlen_kv, :, :] = xk
            self.v_cache[:bsz, start_pos: start_pos + seqlen_kv, :, :] = xv
            keys = self.k_cache[:bsz, :start_pos + seqlen_kv]
            values = self.v_cache[:bsz, :start_pos + seqlen_kv]

        output = flash_attn_func(
            xq, keys, values, dropout_p=0.0, causal=mask is not None)
        output = output.contiguous().view(bsz, seqlen_q, -1)

        return self.wo(output)

    def allocate_kv_cache(self, max_batch_size: int, max_seq_len: int) -> None:
        kv_cache_shape = (max_batch_size, max_seq_len,
                          self.n_local_heads, self.head_dim)
        if self.k_cache is None or self.k_cache.size() != kv_cache_shape:
            self.k_cache = torch.empty(kv_cache_shape)
        if self.v_cache is None or self.v_cache.size() != kv_cache_shape:
            self.v_cache = torch.empty(kv_cache_shape)

    def destroy_kv_cache(self) -> None:
        self.k_cache, self.v_cache = None, None

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * \
            ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=default_linear_init,
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=default_linear_init
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=default_linear_init
        )

    def _silu_gating(self, x, y):
        return F.silu(x) * y

    def forward(self, x):
        return self.w2(self._silu_gating(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def _forward_ffn(self, h):
        return h + self.feed_forward(self.ffn_norm(h))

    def _forward_attention(self, x, start_pos, freqs_cis, mask, prompt):
        return x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, prompt)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], prompt=None):
        h = self._forward_attention(x, start_pos, freqs_cis, mask, prompt)
        out = self._forward_ffn(h)
        return out


class QformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        args = ModelArgs()
        args.n_heads = 8
        args.dim = 1024
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads
        self.attention = Attention(args)
        self.cross_attention = CrossAttention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def _forward_ffn(self, h):
        return h + self.feed_forward(self.ffn_norm(h))

    def _forward_attention(self, x, start_pos, freqs_cis, mask, prompt):
        return x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, prompt)

    def _forward_cross_attention(self, x_q, x_kv, start_pos, freqs_cis, mask, prompt):
        return x_q + self.cross_attention.forward(self.attention_norm(x_q), self.attention_norm(x_kv),start_pos, freqs_cis, mask, prompt)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], prompt=None):

        # learnable queries self_attention
        x_q_h = self._forward_attention(x_q, start_pos, freqs_cis, mask, prompt)
        # x_q_h_2 = self._forward_ffn(x_q_h)
        # cross attention
        h = self._forward_cross_attention(x_q_h, x_kv, start_pos, freqs_cis, mask, prompt)
        out = self._forward_ffn(h)

        return out
    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
    
        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=nn.init.normal_,
        )
        
        # The instance of LLaMA7B
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        # added by zch, frozen LLaMA2 7B. In the original code, the LLaMA 7B is not frozen.
        # Now, when use action loss, we try to unfrozen the parameters.
        for param in self.layers.parameters():
            param.requires_grad = False
            # param.data = param.data.half()

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
 
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=default_linear_init,
        )

        # here 27 is for mmfi, in the following, set it to a params setting parameter
        self.output_action = ColumnParallelLinear(
            params.dim, 27, bias=False, init_method=default_linear_init,
        )

        # The complex positional embeddings.
        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        # load clip ViT-L-14 and frozen its parameters.
        # check here to see if self.clip is load with pretrained parameters -- zch.
        self.clip, _, _ = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai')
        for param in self.clip.parameters():
            param.requires_grad = False
            param.data = param.data.half()  # convert to half precision, float16
        self.clip.transformer = None

        self.cache_image_words = 0  # for inference

        clip_width = self.clip.visual.conv1.out_channels 
        self.clip_positional_embeddings = SinusoidalPositionalEmbedding(dim=clip_width)
        # the MLP projector change to Qformer
        # self.proj = nn.Sequential(
        #     nn.GELU(),
        #     nn.Linear(clip_width, clip_width),
        # )


        # self.connector = QformerBlock()

        self.num_ca = 8
        self.connector = torch.nn.ModuleList()
        for i in range(self.num_ca):
            self.connector.append(QformerBlock())

        # resampler_params = copy.deepcopy(params)
        # resampler_params.n_heads = 16
        # resampler_params.dim = clip_width
        # self.connector = TransformerBlock(0, resampler_params)
    
        self.conv1 = nn.ModuleDict()  # conv2D for each modality
        self.modality_encoders = nn.ModuleDict()
        self.modality_linears = nn.ModuleDict()

        self.clip_proj1 = nn.ModuleDict()
        self.clip_proj2 = nn.ModuleDict()

        self.start_tag = nn.ParameterDict()
        self.end_tag = nn.ParameterDict()

        self.modals = ['video', 'depth', 'wifi', 'mmwave', 'lidar', 'infra', 'rfid']
        
      
        for modal in self.modals:
    
            
            if modal in ['video','depth', 'infra']:
   
                resnet18 = models.resnet18(pretrained=True)
                self.modality_encoders[modal] = nn.Sequential(*list(resnet18.children())[:-2])

                # load pretrained parameters
                if modal == "video":
                    # self.convnet_conv_rgb = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1, stride=1)
                    pretrained_path = "./modality_specific_encoders/mmfi_encoders/cs_pretrained_encoders/rgb_network.pth"
                elif modal == "depth":
                    # self.convnet_conv_depth = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1, stride=1)
                    pretrained_path = "./modality_specific_encoders/mmfi_encoders/cs_pretrained_encoders/depth_network.pth"
                else:
                    pretrained_path = "./modality_specific_encoders/xrf55_encoders/cs_pretrained_encoders/infra_network.pth"

                # load pretrained encoders
                checkpoint = torch.load(pretrained_path)
                encoder_state_dict = self.modality_encoders[modal].state_dict()
                for k, v in encoder_state_dict.items():
                    checkpoint_name = "encoder." + k
                    encoder_state_dict[k] = checkpoint[checkpoint_name]

                msg = self.modality_encoders[modal].load_state_dict(encoder_state_dict)
                print(msg)
    
                self.modality_linears[modal] = nn.Linear(512, clip_width)

              
                for param in self.modality_encoders[modal].parameters():
                    param.requires_grad = False

            elif modal == 'lidar':
                # self.convnet_conv_lidar = nn.Conv1d(in_channels=1024, out_channels=64, kernel_size=1, stride=1)
                from model.lib.point_utils import MMFiPointPatchEmbed
                self.conv1[modal] = MMFiPointPatchEmbed(
                    in_channels=3, channels=clip_width, sample_number=1024)

                self.modality_encoders[modal] = PointNetEncoder(input_dim=3)
                pretrained_path = "./modality_specific_encoders/mmfi_encoders/cs_pretrained_encoders/lidar_network.pth"

                checkpoint = torch.load(pretrained_path)
                encoder_state_dict = self.modality_encoders[modal].state_dict()
                for k, v in encoder_state_dict.items():
                    checkpoint_name = "encoder." + k
                    encoder_state_dict[k] = checkpoint[checkpoint_name]

                msg = self.modality_encoders[modal].load_state_dict(encoder_state_dict)
                print(msg)

                self.modality_linears[modal] = nn.Linear(1024, clip_width)

                for param in self.modality_encoders[modal].parameters():
                    param.requires_grad = False

            elif modal == 'mmwave':
                # self.convnet_conv_mmwave = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1, stride=1)
                from model.lib.point_utils import MMFiPointPatchEmbed
                self.conv1[modal] = MMFiPointPatchEmbed(
                    in_channels=5, channels=clip_width, sample_number=64)
            
                self.modality_encoders[modal] = PointNetEncoder(input_dim=5)
                pretrained_path = "./modality_specific_encoders/mmfi_encoders/cs_pretrained_encoders/mmwave_network.pth"

                checkpoint = torch.load(pretrained_path)
                encoder_state_dict = self.modality_encoders[modal].state_dict()
                for k, v in encoder_state_dict.items():
                    checkpoint_name = "encoder." + k
                    encoder_state_dict[k] = checkpoint[checkpoint_name]

                msg = self.modality_encoders[modal].load_state_dict(encoder_state_dict)
                print(msg)

                self.modality_linears[modal] = nn.Linear(1024, clip_width)

                for param in self.modality_encoders[modal].parameters():
                    param.requires_grad = False

            elif modal == 'wifi':
                # self.convnet_conv_wifi = nn.Conv1d(in_channels=56*4, out_channels=32, kernel_size=1, stride=1)
                self.conv1[modal] = nn.Conv2d(3, clip_width, kernel_size=(3, 3), stride=(2, 2))
                
                # resnet18 = models.resnet18(pretrained=True)
                # self.modality_encoders[modal] = nn.Sequential(*list(resnet18.children())[:-2])
                self.modality_encoders[modal] = WiFi_Encoder()
                
                pretrained_path = "./modality_specific_encoders/mmfi_encoders/cs_pretrained_encoders/wifi_network.pth"
                checkpoint = torch.load(pretrained_path)
                encoder_state_dict = self.modality_encoders[modal].state_dict()
                for k, v in encoder_state_dict.items():
                    encoder_state_dict[k] = checkpoint[k]

                msg = self.modality_encoders[modal].load_state_dict(encoder_state_dict)
                print(msg)

                self.modality_linears[modal] = nn.Linear(512, clip_width)

                for param in self.modality_encoders[modal].parameters():
                    param.requires_grad = False

            elif modal == 'rfid':
                self.conv1[modal] = nn.Conv2d(23, clip_width, kernel_size=1, stride=1)
                
                # resnet18 = models.resnet18(pretrained=True)
                # self.modality_encoders[modal] = nn.Sequential(*list(resnet18.children())[:-2])
                # self.convnet_conv_rfid = nn.Conv1d(in_channels=148, out_channels=32, kernel_size=1, stride=1)
                self.modality_encoders[modal] = RFID_Encoder()
                
                pretrained_path = "./modality_specific_encoders/xrf55_encoders/cs_pretrained_encoders/rfid_network.pth"
                checkpoint = torch.load(pretrained_path)
                encoder_state_dict = self.modality_encoders[modal].state_dict()
                for k, v in encoder_state_dict.items():
                    encoder_state_dict[k] = checkpoint[k]

                msg = self.modality_encoders[modal].load_state_dict(encoder_state_dict)
                print(msg)

                self.modality_linears[modal] = nn.Linear(512, clip_width)

                for param in self.modality_encoders[modal].parameters():
                    param.requires_grad = False

            # self.clip_proj1[modal] = nn.Sequential(
            #     nn.Linear(clip_width, clip_width),
            #     nn.GELU(),
            #     nn.Linear(clip_width, clip_width),
            #     nn.LayerNorm(clip_width))

            self.clip_proj1[modal] = nn.Sequential(
                nn.Linear(clip_width, clip_width),
                nn.LayerNorm(clip_width))

            self.clip_proj2[modal] = nn.Sequential(
                nn.Linear(clip_width, params.dim),
                nn.LayerNorm(params.dim))

            self.start_tag[modal] = nn.Parameter(torch.rand(1, 1, params.dim))
            self.end_tag[modal] = nn.Parameter(torch.rand(1, 1, params.dim))
        
        # TODO: Freeze some parameters at here. Freeze LLM for pretraining and Projection for finetuining.

    # @torch.no_grad()

    def clip_encode_image(self, x, modal='video'):
        # Here is the CLIP, the modality-aware prefix-tuning should be added here. -- zch

        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                      x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        pos_embedding = self.clip_positional_embeddings(x)

        x = x + pos_embedding.to(x.dtype) 
      
        x = self.clip.visual.ln_pre(x)  

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # preserve all spatial tokens
        x = self.clip.visual.ln_post(x[:, :, :])

        # if self.clip.visual.proj is not None:
        #    x = x @ self.clip.visual.proj

        return x

    def encode_image(self, x, modal='video'):
        bsz = x.size(0)
        T = 1

        if modal == 'video':
            # just copy the video modality
            B, T = x.shape[:2]
            bsz = B * T
            x_tmp = x.reshape(bsz, *x.shape[2:])
            x_conv = self.clip.visual.conv1(x_tmp).view(x_tmp.shape[0],1024,-1).permute(0,2,1)  # [15, 1024, 16, 16, 3]

            x_token = x_conv
            x_convnet = self.modality_encoders[modal](x_tmp)
            x_convnet = x_convnet.view(x_convnet.shape[0], -1, x_convnet.shape[1])
            x_convnet = self.modality_linears[modal](x_convnet)

            
        elif modal == 'depth':
            B, T = x.shape[:2]
            bsz = B * T
            x_tmp = x.reshape(bsz, *x.shape[2:])
            x_tmp = x_tmp.repeat(1,3,1,1)

            x_conv = self.clip.visual.conv1(x_tmp).view(x_tmp.shape[0],1024,-1).permute(0,2,1)  # [15, 1024, 16, 16, 3]

            x_token = x_conv
            x_convnet = self.modality_encoders[modal](x_tmp)
            x_convnet = x_convnet.view(x_convnet.shape[0], -1, x_convnet.shape[1])
            x_convnet = self.modality_linears[modal](x_convnet)

            
        elif modal == 'infra':
            B, T = x.shape[:2]
            bsz = B * T
            x_tmp = x.reshape(bsz, *x.shape[2:])
            x_tmp = x_tmp.repeat(1,3,1,1)
            x_conv = self.clip.visual.conv1(x_tmp).view(x_tmp.shape[0],1024,-1).permute(0,2,1)  # [15, 1024, 16, 16, 3]

            x_token = x_conv
            x_convnet = self.modality_encoders[modal](x_tmp)
            x_convnet = x_convnet.view(x_convnet.shape[0], -1, x_convnet.shape[1])
            x_convnet = self.modality_linears[modal](x_convnet)

        elif modal == 'lidar':
       
            B, T = x.shape[:2]
            bsz = B * T
            if not self.training:
                x = x.squeeze()
            x_tmp = x.reshape(bsz, *x.shape[2:])

            x_token, x_selected = self.conv1[modal](x_tmp)
            x_token = x_token.to(x_tmp.dtype).squeeze().permute(0,2,1) 
            x_selected = x_selected.squeeze()
            x_convnet = self.modality_encoders[modal](x_selected)
            x_convnet = x_convnet.to(x_tmp.dtype)
            x_convnet = x_convnet.permute(0,2,1)
            x_convnet = self.modality_linears[modal](x_convnet)

        elif modal == 'mmwave':
            # [B, 64, 5] -> [B, 1024, 64, 1]
            B, T = x.shape[:2]
            bsz = B * T
            if not self.training:
                x = x.squeeze()

            x_tmp = x.reshape(bsz, *x.shape[2:])

            x_token, x_selected = self.conv1[modal](x_tmp)
            x_token = x_token.to(x_tmp.dtype).squeeze().permute(0,2,1)  
            x_selected = x_selected.squeeze()
            x_convnet = self.modality_encoders[modal](x_selected)
            x_convnet = x_convnet.to(x_tmp.dtype)
            x_convnet = x_convnet.permute(0,2,1)
        
            x_convnet = self.modality_linears[modal](x_convnet)

        elif modal == 'wifi':

            B, T = x.shape[:2]
            bsz = B * T
            if not self.training:
                x = x.squeeze()

            x_tmp = x.reshape(bsz, *x.shape[2:])
            if x_tmp.shape[1] == 3:
                x_token = self.conv1[modal](x_tmp).view(x_tmp.shape[0],1024,-1).permute(0,2,1)
            if x_tmp.shape[1] == 9:
                x_1 = x_tmp[:,0:3,:,:]
                x_2 = x_tmp[:,3:6,:,:]
                x_3 = x_tmp[:,6:9,:,:]
                x_token_1 = self.conv1[modal](x_1).view(x_1.shape[0],1024,-1).permute(0,2,1).unsqueeze(dim=1) 
                x_token_2 = self.conv1[modal](x_2).view(x_2.shape[0],1024,-1).permute(0,2,1).unsqueeze(dim=1)  
                x_token_3 = self.conv1[modal](x_3).view(x_3.shape[0],1024,-1).permute(0,2,1).unsqueeze(dim=1) 
                x_token = torch.cat((x_token_1, x_token_2, x_token_3), dim=1).mean(dim=1)

            x_conv_tmp = x_tmp.view(B, T, x_tmp.shape[1], x_tmp.shape[2], x_tmp.shape[3]) 
            x_convnet = self.modality_encoders[modal](x_conv_tmp)
            x_convnet = x_convnet.view(x_convnet.shape[0]*x_convnet.shape[1], -1, x_convnet.shape[2])
            x_convnet = self.modality_linears[modal](x_convnet)
        
        elif modal == 'rfid':
            x = x.unsqueeze(dim=1)
            B, T = x.shape[:2]
            bsz = B * T
            x = x.reshape(bsz, *x.shape[2:])
            if not self.training:
                x = x.squeeze()
            x_tmp = x.unsqueeze(dim=-1)

            x_token = self.conv1[modal](x_tmp).view(x_tmp.shape[0],1024,-1).permute(0,2,1) 
            x_convnet = self.modality_encoders[modal](x_tmp.squeeze()).permute(0,2,1)  # [bz, 5, 512]
            x_convnet = self.modality_linears[modal](x_convnet)

        x_clip = self.clip_encode_image(x_token, modal=modal)  
        # image_feats = torch.cat((x_clip, x_convnet), dim=1)
        x_clip_global = x_clip[:,0,:].unsqueeze(dim=1)
        x_clip = x_clip[:,1:,:]  # remove the visual cls embeddings

        # output_channels = 32
        if modal == "video":
            # x_clip = x_clip.view(x_clip.shape[0], 16, 16, x_clip.shape[2])
            # target_size = (16 // 2, 16 // 2)
            # image_feats = self.bilinear_interpolation(x_clip, target_size)
            # image_feats = self.convnet_conv_rgb(x_clip)
            image_feats = F.adaptive_avg_pool1d(x_clip.permute(0,2,1), 64).permute(0, 2, 1) 
            image_feats = torch.cat((x_clip_global, image_feats), dim=1) 

        elif modal == "depth":
            # x_clip = x_clip.view(x_clip.shape[0], 16, 16, x_clip.shape[2])
            # target_size = (16 // 2, 16 // 2)
            # image_feats = self.bilinear_interpolation(x_clip, target_size) 
            image_feats = F.adaptive_avg_pool1d(x_clip.permute(0,2,1), 64).permute(0, 2, 1) 
            image_feats = torch.cat((x_clip_global, image_feats), dim=1) 

        elif modal == "infra":
            x_clip = x_clip.view(x_clip.shape[0], 16, 16, x_clip.shape[2])
            target_size = (16 // 2, 16 // 2)
            image_feats = self.bilinear_interpolation(x_clip, target_size) 
            image_feats = torch.cat((x_clip_global, image_feats), dim=1)  
        # point cloud we use 1D conv to reduce channel
        elif modal == "mmwave":
            # x_clip = x_clip.view(x_clip.shape[0], 8, 8, x_clip.shape[2])
            # target_size = (8 // 2, 8 // 2)
            # image_feats = self.bilinear_interpolation(x_clip, target_size)
            image_feats = F.adaptive_avg_pool1d(x_clip.permute(0,2,1), 64).permute(0, 2, 1) 
            image_feats = torch.cat((x_clip_global, image_feats), dim=1) 

        elif modal == "lidar":
            # x_clip = x_clip.view(x_clip.shape[0], 32, 32, x_clip.shape[2])
            # target_size = (32 // 4, 32 // 4)
            # image_feats = self.bilinear_interpolation(x_clip, target_size) 
            image_feats = F.adaptive_avg_pool1d(x_clip.permute(0,2,1), 128).permute(0, 2, 1) 
            image_feats = torch.cat((x_clip_global, image_feats), dim=1) 

        elif modal == "wifi":
            # x_clip = x_clip.view(x_clip.shape[0], 56, 4, x_clip.shape[2])
            # x_clip = x_clip.view(x_clip.shape[0], 77, 18, x_clip.shape[2])
            # target_size = (56 // 4, 4 // 2)
            # image_feats = self.bilinear_interpolation(x_clip, target_size)
            image_feats = F.adaptive_avg_pool1d(x_clip.permute(0,2,1), 16).permute(0, 2, 1) 
            image_feats = torch.cat((x_clip_global, image_feats), dim=1) 

        elif modal == "rfid":
            # x_clip = x_clip.view(x_clip.shape[0], 4, 37, x_clip.shape[2])
            image_feats = self.convnet_conv_rfid(x_clip)
            # image_feats = x_clip.mean(dim=1)
            image_feats = torch.cat((x_clip_global, image_feats), dim=1) 

        # image_feats = x_clip
        
        bsz = int(bsz / T)
        image_feats = image_feats.reshape(
            bsz, T, *image_feats.shape[1:]).mean(dim=1)

        x_convnet = x_convnet.reshape(
            bsz, T, *x_convnet.shape[1:]).mean(dim=1)
        
        image_feats = self.clip_proj1[modal](image_feats)  # just a linear project

        query_feat = self.connector[0](image_feats, x_convnet, 0, None, None)

        for i in range(1, self.num_ca):  # the following ca layer
            query_feat = self.connector[i](query_feat, x_convnet, 0, None, None)
        
        # In our baseline, directly utilize a projector, no resample tokens.
        image_feats = self.clip_proj2[modal](query_feat)

        return image_feats

    def forward(self, examples, image=None, modal='image'):
        self._destroy_kv_cache()  # training always disables kv cache
        # modal = modal[0]
        modal = modal[0].split("_")[-1]  # xrf55_wifi, mmfi_wifi
        _bsz, seqlen = examples.shape
        h = self.tok_embeddings(examples)
        self.freqs_cis = self.freqs_cis.to(h.device)

        start_pos = 0
        prefix_len = 0
        if image is not None:
            # bos = "Beginning of Sequence"
            h_bos, h_caption = h[:, :1], h[:, 1:]  # h_caption = [4, 2017, 4096], the size is related to LLaMA2
            # here is the Tokenizer in Paper, should be changed for mmfi and xrf dataset, or we unify them.
            # the design of adapter, also needed in here? the Tokenizer is inside the encode_image, e.g., self.conv1
            image_tokens = self.encode_image(image, modal)  
            

            h = torch.cat((h_bos, self.start_tag[modal].expand(
                _bsz, -1, -1), image_tokens, self.end_tag[modal].expand(_bsz, -1, -1), h_caption), dim=1)
            # bos + image token + start_tag[modal], end_tag[modal] is used for caption generation
            prefix_len = image_tokens.shape[1] + 1 + 1
            seqlen = h.shape[1]
        # the freqs_cis is a kind of positional embeddings!
        # specifically, cis is the representation of complex values: cis(θ)=cos(θ)+isin(θ)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        for layer in self.layers:  # here is the LLaMA2 layers
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)

        output = self.output(h[:, prefix_len:, :])  
        # output_action = self.output_action(h[:, :prefix_len, :].mean(dim=1)) 
        output_action = self.output_action(h[:, 2:prefix_len, :].mean(dim=1)) 

        return output, output_action

    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor, start_pos: int, image=None, modal='image'):
        modal = modal[0] if isinstance(modal, list) else modal
        _bsz, seqlen = tokens.shape
        if start_pos == 0:
            # kv cache will not re-allocate if size is unchanged
            self._allocate_kv_cache(_bsz)
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)

        if image is not None:
            h_bos, h_caption = h[:, :1], h[:, 1:]
            image_tokens = self.encode_image(image, modal)

            self.cache_image_words = image_tokens.shape[1]
            h = torch.cat((h_bos, self.start_tag[modal].repeat(_bsz, 1, 1), image_tokens, self.end_tag[modal].repeat(_bsz, 1, 1), h_caption), dim=1)
            seqlen = h.shape[1]
            freqs_cis = self.freqs_cis[0: seqlen]
        else:
            if start_pos == 0:
                self.cache_image_words = 0
                freqs_cis = self.freqs_cis[0: seqlen]
            else:
                # if image was not None when start_pos=0,
                # the offset should be added to start_pos within later forward_inference calls
                start_pos = start_pos + self.cache_image_words
                freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        # freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()

    @torch.inference_mode()
    def forward_action_inference(self, tokens: torch.Tensor, start_pos: int, image=None, modal='image'):
        modal = modal[0] if isinstance(modal, list) else modal
        _bsz, seqlen = tokens.shape
        if start_pos == 0:
            # kv cache will not re-allocate if size is unchanged
            self._allocate_kv_cache(_bsz)
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)

        if image is not None:
            h_bos, h_caption = h[:, :1], h[:, 1:]
            image_tokens = self.encode_image(image, modal)
            self.cache_image_words = image_tokens.shape[1]
            h = torch.cat((h_bos, self.start_tag[modal].repeat(_bsz, 1, 1), image_tokens, self.end_tag[modal].repeat(_bsz, 1, 1), h_caption), dim=1)
            prelen = 1 + image_tokens.shape[1] + 1
            seqlen = h.shape[1]
            freqs_cis = self.freqs_cis[0: seqlen]
        else:
            if start_pos == 0:
                self.cache_image_words = 0
                freqs_cis = self.freqs_cis[0: seqlen]
            else:
                # if image was not None when start_pos=0,
                # the offset should be added to start_pos within later forward_inference calls
                start_pos = start_pos + self.cache_image_words
                freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        # freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        # output = self.output(h[:, -1, :])  # only compute last logits
        image_tokens = h[:,2:prelen,:]  # obtain image token, exclude prefix, start and end
        output_action = self.output_action(image_tokens.mean(dim=1))
        return output_action.float()

    def bilinear_interpolation(self, input_tensor, target_size):

        tensor_downsampled = F.interpolate(
            input_tensor.float().permute(0, 3, 1, 2),  # Convert to [B, C, H, W]
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)  # Convert back to [B, H, W, C]
        tensor_downsampled = tensor_downsampled.to(input_tensor.dtype)
        tensor_downsampled = tensor_downsampled.view(tensor_downsampled.shape[0], tensor_downsampled.shape[1]*tensor_downsampled.shape[2], tensor_downsampled.shape[3])

        return tensor_downsampled


    def extract_features(self, x, modal='mmfi_video'):
        bsz = x.size(0)
        T = 1

        if modal == 'mmfi_video':
            # just copy the video modality
            B, T = x.shape[:2]
            bsz = B * T
            # pos_embedding = self.clip.visual.positional_embedding[1:]  # remove cls visual embeddings

            x_tmp = x.reshape(bsz, *x.shape[2:])
            x_conv = self.clip.visual.conv1(x_tmp).view(x_tmp.shape[0],1024,-1).permute(0,2,1)  # [15, 1024, 16, 16, 3]

            x_token = x_conv
            x_convnet = self.modality_encoders[modal](x_tmp)
            x_convnet = x_convnet.view(x_convnet.shape[0], -1, x_convnet.shape[1])
            x_convnet = self.modality_linears[modal](x_convnet)

            
            
        elif modal == 'mmfi_depth':
            B, T = x.shape[:2]
            bsz = B * T

            x_tmp = x.reshape(bsz, *x.shape[2:])

            x_tmp = x_tmp.repeat(1,3,1,1).float()
            if not self.training:
                x_tmp = x_tmp.half()

            x_conv = self.clip.visual.conv1(x_tmp).view(x_tmp.shape[0],1024,-1).permute(0,2,1)  

            x_token = x_conv
            x_convnet = self.modality_encoders[modal](x_tmp)
            x_convnet = x_convnet.view(x_convnet.shape[0], -1, x_convnet.shape[1])
            x_convnet = self.modality_linears[modal](x_convnet)

        elif modal == 'mmfi_lidar':

            B, T = x.shape[:2]
            bsz = B * T
            if not self.training:
                x = x.squeeze()
            x_tmp = x.reshape(bsz, *x.shape[2:])

            if self.training:
                x_token, x_selected = self.conv1[modal](x_tmp.float())
                x_token = x_token.to(x_tmp.dtype).squeeze().permute(0,2,1) 
                x_selected = x_selected.squeeze()
                x_convnet = self.modality_encoders[modal](x_selected)
                x_convnet = x_convnet.to(x_tmp.dtype)
                x_convnet = x_convnet.permute(0,2,1)  

            else:
                x_token, x_selected = self.conv1[modal](x_tmp.half())
                x_token = x_token.to(x_tmp.dtype).squeeze().permute(0,2,1)
                x_selected = x_selected.squeeze()
                x_convnet = self.modality_encoders[modal](x_selected)
                x_convnet = x_convnet.to(x_tmp.dtype)
                x_convnet = x_convnet.permute(0,2,1)
            
            x_convnet = self.modality_linears[modal](x_convnet)
        elif modal == 'mmfi_mmwave':
            # [B, 64, 5] -> [B, 1024, 64, 1]
  
            B, T = x.shape[:2]
            bsz = B * T
            if not self.training:
                x = x.squeeze()

            x_tmp = x.reshape(bsz, *x.shape[2:])

            if self.training:
                x_token, x_selected = self.conv1[modal](x_tmp.float())
                x_token = x_token.to(x_tmp.dtype).squeeze().permute(0,2,1)  
                x_selected = x_selected.squeeze()
                x_convnet = self.modality_encoders[modal](x_selected)
                x_convnet = x_convnet.to(x_tmp.dtype)
                x_convnet = x_convnet.permute(0,2,1)
            else:
                x_token, x_selected = self.conv1[modal](x_tmp.half())
                x_token = x_token.to(x_tmp.dtype).squeeze().permute(0,2,1)
                x_selected = x_selected.squeeze()
                x_convnet = self.modality_encoders[modal](x_selected)
                x_convnet = x_convnet.to(x_tmp.dtype)
                x_convnet = x_convnet.permute(0,2,1)

            x_convnet = self.modality_linears[modal](x_convnet)
            
        elif modal == 'mmfi_infra':
            pass


        elif modal == 'mmfi_wifi':
            B, T = x.shape[:2]
            bsz = B * T
            if not self.training:
                x = x.squeeze()

            x_tmp = x.reshape(bsz, *x.shape[2:])

            if self.training:
                x_tmp = x_tmp.float()
            else:
                x_tmp = x_tmp.half()

            x_token = self.conv1[modal](x_tmp).view(x_tmp.shape[0],1024,-1).permute(0,2,1)
            x_conv_tmp = x_tmp.view(B, T, x_tmp.shape[1], x_tmp.shape[2], x_tmp.shape[3]) 
            x_convnet = self.modality_encoders[modal](x_conv_tmp)
            x_convnet = x_convnet.view(x_convnet.shape[0]*x_convnet.shape[1], -1, x_convnet.shape[2])
            x_convnet = self.modality_linears[modal](x_convnet)

        x_clip = self.clip_encode_image(x_token, modal=modal)  
        # image_feats = torch.cat((x_clip, x_convnet), dim=1)
        x_clip_global = x_clip[:,0,:].unsqueeze(dim=1)
        x_clip = x_clip[:,1:,:]  # remove the visual cls embeddings

        if modal == "mmfi_video":
            x_clip = x_clip.view(x_clip.shape[0], 16, 16, x_clip.shape[2])
            target_size = (16 // 2, 16 // 2)
            image_feats = self.bilinear_interpolation(x_clip, target_size)
            image_feats = torch.cat((x_clip_global, image_feats), dim=1) 

        elif modal == "mmfi_depth":
            x_clip = x_clip.view(x_clip.shape[0], 16, 16, x_clip.shape[2])
            target_size = (16 // 2, 16 // 2)
            image_feats = self.bilinear_interpolation(x_clip, target_size) 
            image_feats = torch.cat((x_clip_global, image_feats), dim=1) 
        
        elif modal == "mmfi_mmwave":
            x_clip = x_clip.view(x_clip.shape[0], 8, 8, x_clip.shape[2])
            target_size = (8 // 2, 8 // 2)
            image_feats = self.bilinear_interpolation(x_clip, target_size)
            image_feats = torch.cat((x_clip_global, image_feats), dim=1) 

        elif modal == "mmfi_lidar":
            x_clip = x_clip.view(x_clip.shape[0], 32, 32, x_clip.shape[2])
            target_size = (32 // 4, 32 // 4)
            image_feats = self.bilinear_interpolation(x_clip, target_size) 
            image_feats = torch.cat((x_clip_global, image_feats), dim=1) 

        elif modal == "mmfi_wifi":
            x_clip = x_clip.view(x_clip.shape[0], 56, 4, x_clip.shape[2])
            target_size = (56 // 2, 4 // 2)
            image_feats = self.bilinear_interpolation(x_clip, target_size)
            image_feats = torch.cat((x_clip_global, image_feats), dim=1) 

        bsz = int(bsz / T)
        image_feats = image_feats.reshape(
            bsz, T, *image_feats.shape[1:]).mean(dim=1)

        x_convnet = x_convnet.reshape(
            bsz, T, *x_convnet.shape[1:]).mean(dim=1)
        
        image_feats_2 = self.clip_proj1[modal](image_feats)  # just a linear project

        query_feat = self.connector[0](image_feats_2, x_convnet, 0, None, None)

        for i in range(1, self.num_ca):  # the following ca layer
            query_feat = self.connector[i](query_feat, x_convnet, 0, None, None)
        
        # In our baseline, directly utilize a projector, no resample tokens.
        image_feats_2 = self.clip_proj2[modal](query_feat)

        return image_feats, query_feat, image_feats_2
    
    @torch.inference_mode()
    def forward_extract_features(self, tokens, image=None, modal='image'):
        modal = modal[0] if isinstance(modal, list) else modal
        h = self.tok_embeddings(tokens)
        h_bos, h_caption = h[:, :1], h[:, 1:]
        before_connector_feats, after_connector_feats, llm_image_feats = self.extract_features(image, modal)
            
        return before_connector_feats, after_connector_feats, llm_image_feats, h_caption
    

    def _allocate_kv_cache(self, max_batch_size: int) -> None:
        for layer in self.layers:
            layer.attention.allocate_kv_cache(
                max_batch_size, self.params.max_seq_len)

    def _destroy_kv_cache(self) -> None:
        for layer in self.layers:
            layer.attention.destroy_kv_cache()
