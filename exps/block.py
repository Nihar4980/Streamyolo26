import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F                      
import math
from typing import Callable, Optional
from torch.nn.functional import pad
from natten import NeighborhoodAttention2D
from natten.functional import attention

class PatchEmbed(nn.Module):
    """
    From Swin Transformer
    https://arxiv.org/abs/2103.14030
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, patch_size=8, in_chans=3, embed_dim=256,flatten_embedding=False):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = (nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
                    if flatten_embedding 
                    else
                    Conv(in_chans, embed_dim, k=self.patch_size, s=self.patch_size,act=False))
        self.norm = nn.LayerNorm(embed_dim) if flatten_embedding else None
        self.flatten_embedding = flatten_embedding

    def forward(self, x):
        B, C, H, W = x.shape
        if W % self.patch_size != 0:
            x = pad(x, (0, self.patch_size - W % self.patch_size))
        if H % self.patch_size != 0:
            x = pad(x, (0, 0, 0, self.patch_size - H % self.patch_size))
        x = self.proj(x)
        if self.flatten_embedding:
            x = x.flatten(2).transpose(1, 2)  # B H W C
            x = self.norm(x)
        return x


# ---------------------------------------------------------
# Neighborhood Attention (local self-attention)
# ---------------------------------------------------------
class NeighborhoodAttention(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=8,
        qkv_bias=False,
        proj_bias=True,
        proj_drop=0.0,
        flatten_embedding=False,
    ):
        super().__init__()

        assert channels % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.flatten_embedding = flatten_embedding

        if flatten_embedding:
            self.norm = nn.LayerNorm(channels)

            self.qkv = nn.Linear(
                channels,
                3 * channels,
                bias=qkv_bias
            )

            self.proj = nn.Linear(
                channels,
                channels,
                bias=proj_bias
            )

        else:
            self.norm = nn.BatchNorm2d(channels)

            self.qkv = Conv(
                channels,
                3 * channels,
                1,
                act=False,
                bias=qkv_bias
            )

            self.proj = Conv(
                channels,
                channels,
                1,
                act=False,
                bias=proj_bias
            )

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        heads = self.num_heads
        dim = self.head_dim

        if self.flatten_embedding:

            B, HW, C = x.shape

            x = self.norm(x)

            qkv = self.qkv(x)
            qkv = qkv.reshape(B, HW, 3, heads, dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)

            # (3, B, heads, HW, dim)

        else:

            B, C, H, W = x.shape

            x = self.norm(x)

            qkv = self.qkv(x)
            qkv = qkv.reshape(B, 3, heads, dim, H * W)
            qkv = qkv.permute(1, 0, 2, 4, 3)

            # (3, B, heads, HW, dim)

            HW = H * W

        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * (dim ** -0.5)

        out = attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            run_persistent_kernel=True
        )

        if self.flatten_embedding:
            out = out.permute(0, 2, 1, 3).reshape(B, HW, C)
        else:
            out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        drop: float = 0.0,
        bias: bool = True,
        flatten_embedding=False
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.layer = (nn.Sequential(nn.Linear(in_features, hidden_features, bias=bias),
                                 nn.GELU(),
                                 nn.Linear(hidden_features, out_features, bias=bias),
                                 nn.Dropout(drop)
                                ) 
                    if flatten_embedding else
                    nn.Sequential(Conv(in_features, hidden_features, 1, bias=bias), 
                                  Conv(hidden_features, out_features,1, act=False, bias=bias),
                                  nn.Dropout(drop))
                   )       

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
# ---------------------------------------------------------
# NAT Block (Attention + MLP)
# ---------------------------------------------------------
class NATBlock(nn.Module):
    """Neighborhood Attention + MLP (Transformer-like block)"""
    def __init__(self,
                dim: int=256,
                num_heads: int=8,
                mlp_ratio: float = 4.0,
                qkv_bias: bool = False,
                proj_bias: bool = True,
                ffn_bias: bool = True,
                drop: float = 0.0,
                proj_drop: float = 0.0,
                shortcut: bool = True,
                flatten_embedding=False
            ): 
        super().__init__()
        self.attn = NeighborhoodAttention(
                                            channels=dim, 
                                            num_heads=num_heads,
                                            qkv_bias=False,
                                            proj_bias=True, 
                                            proj_drop=0.0,
                                            flatten_embedding=flatten_embedding
                                         )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
                        in_features=dim,
                        hidden_features=mlp_hidden_dim,
                        out_features= dim,
                        drop=drop,
                        bias=ffn_bias,
                        flatten_embedding=flatten_embedding
                      )
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.mlp(x) if self.add else self.ffn(x)
        return x


# ---------------------------------------------------------
# Combined PatchEmbed + NATBlock class
# ---------------------------------------------------------
class NAVisionTransformer(nn.Module):

    def __init__(
        self,
        #input_size,
        patch_size=8,
        in_chans=3,
        embed_dim=256,
        num_heads=8,
        depth = 2,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        proj_drop: float = 0.0,
        shortcut: bool = True,
        flatten_embedding=False

    ):
        super().__init__()
        # Patch Embedding
        self.patch_embed = PatchEmbed(
                                        patch_size=patch_size,
                                        in_chans=in_chans,
                                        embed_dim=embed_dim,
                                        flatten_embedding=flatten_embedding,
                                    )
        
        # self.num_patches=(input_size[0]//patch_size)*(input_size[1]//patch_size)
        # self.positional_embedding=nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.flatten_embedding=flatten_embedding
        self.blocks = nn.ModuleList(
            [
                NATBlock(
                            dim = embed_dim,
                            num_heads = num_heads,
                            mlp_ratio= mlp_ratio,
                            qkv_bias = qkv_bias,
                            proj_bias=proj_bias,
                            ffn_bias= ffn_bias,
                            drop = drop,
                            proj_drop = proj_drop,
                            shortcut=shortcut,
                            flatten_embedding=flatten_embedding,
                        )
                for i in range(depth)
            ]
        )
        self.patch_size = patch_size
        self.embed_dim = embed_dim
    
    def forward(self, x):
        B,C,H,W = x.shape
        Hp,Wp = H//self.patch_size, W//self.patch_size
        # Step 1: Patch Embedding
        x = self.patch_embed(x)
        # if self.flatten_embedding:
        #     x+=self.positional_embedding
        # Step 2: NATBlock
        for blk in self.blocks:
            x = blk(x)
        x = x.reshape(B,Hp,Wp,self.embed_dim).permute(0,3,1,2) if self.flatten_embedding  else x
        return x

class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True,bias=False):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        padding = p if p is not None else d * (k - 1) // 2
        self.conv = nn.Conv2d(c1, c2, k, s, padding=padding, groups=g, dilation=d, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
           
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


# =====================================================
# NAT (Neighborhood Attention)
# =====================================================
class DynamicNA2D(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=4,
        kernel_size=(7, 7),
        stride=(1, 1),
        dilation=(1, 1),
        is_causal=False,
    ):
        super().__init__()

        self.attn = NeighborhoodAttention2D(
            embed_dim=channels,
            num_heads=num_heads,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # BCHW → BHWC
        x = self.attn(x)
        x = x.permute(0, 3, 1, 2)
        return x


# ---------------------------------------------------------
# Transformer-style Feed Forward Network (MLP)
# ---------------------------------------------------------
class MLP(nn.Module):
    """Local MLP block with depthwise conv"""
    def __init__(self, dim, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.layer = nn.Sequential(Conv(dim, hidden_dim, 1), 
                      Conv(hidden_dim,dim,1, act=False),
                      nn.Dropout(drop))

    def forward(self, x):
        x = self.layer(x) 
        return x 


# =====================================================
# NAT Block (Attention + MLP)
# =====================================================
class NATransfomerBlock(nn.Module):
    """Neighborhood Attention + MLP (Transformer-like block)"""
    def __init__(self,
                dim: int=256,
                num_heads: int=4,
                kernel_size=(7, 7),
                stride=(1, 1),
                dilation=(1, 1),
                is_causal=False,
                mlp_ratio: float = 2.0,
                drop: float = 0.0,
                shortcut: bool = True,
            ): 
        super().__init__()
        self.attn = DynamicNA2D(
                                    channels = dim,
                                    num_heads=num_heads,
                                    kernel_size = kernel_size,
                                    stride=stride,
                                    dilation=dilation,
                                    is_causal=False
                                            )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
                        dim=dim, mlp_ratio=mlp_ratio, drop=0.0
                      )
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.mlp(x) if self.add else self.ffn(x)
        return x


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3K2_NAattn(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(   self, 
                    dim: int=256,
                    n: int = 2, 
                    e: float = 0.5,
                    num_heads: int=4,
                    kernel_size=(7, 7),
                    stride=(1, 1),
                    dilation=(1, 1),
                    is_causal=False,
                    mlp_ratio: float = 2.0,
                    drop: float = 0.0,
                    shortcut: bool = True,
                    attn = True
                ):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(dim * e)  # hidden channels
        self.cv1 = Conv(dim, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, dim, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
                                
                                    nn.Sequential(
                                        Bottleneck(self.c,self.c,shortcut=shortcut),
                                        NATransfomerBlock(
                                                            dim= self.c,
                                                            num_heads=num_heads,
                                                            kernel_size = kernel_size,
                                                            stride=stride,
                                                            dilation=dilation,
                                                            is_causal=is_causal,
                                                            mlp_ratio = mlp_ratio,
                                                            drop = drop,
                                                            shortcut= shortcut,
                                                         )
                                       
                                                )
                                    if attn
                                    else Bottleneck(self.c,self.c,shortcut=shortcut)
                                    for _ in range(n)
                                )
                            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        #print(self.c, y[0].shape, y[1].shape)
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


        
class AdaHyperedgeGen(nn.Module):
    """
    Generates an adaptive hyperedge participation matrix from a set of vertex features.

    This module implements the Adaptive Hyperedge Generation mechanism. It generates dynamic hyperedge prototypes
    based on the global context of the input nodes and calculates a continuous participation matrix (A)
    that defines the relationship between each vertex and each hyperedge.

    Attributes:
        node_dim (int): The feature dimension of each input node.
        num_hyperedges (int): The number of hyperedges to generate.
        num_heads (int, optional): The number of attention heads for multi-head similarity calculation. Defaults to 4.
        dropout (float, optional): The dropout rate applied to the logits. Defaults to 0.1.
        context (str, optional): The type of global context to use ('mean', 'max', or 'both'). Defaults to "both".

    Methods:
        forward: Takes a batch of vertex features and returns the participation matrix A.

    Examples:
        >>> import torch
        >>> model = AdaHyperedgeGen(node_dim=64, num_hyperedges=16, num_heads=4)
        >>> x = torch.randn(2, 100, 64)  # (Batch, Num_Nodes, Node_Dim)
        >>> A = model(x)
        >>> print(A.shape)
        torch.Size([2, 100, 16])
    """
    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.num_heads = num_heads
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads
        self.context = context

        self.prototype_base = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)
        if context in ("mean", "max"):
            self.context_net = nn.Linear(node_dim, num_hyperedges * node_dim)  
        elif context == "both":
            self.context_net = nn.Linear(2*node_dim, num_hyperedges * node_dim)
        else:
            raise ValueError(
                f"Unsupported context '{context}'. "
                "Expected one of: 'mean', 'max', 'both'."
            )

        self.pre_head_proj = nn.Linear(node_dim, node_dim)
    
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, X):
        B, N, D = X.shape
        if self.context == "mean":
            context_cat = X.mean(dim=1)          
        elif self.context == "max":
            context_cat, _ = X.max(dim=1)          
        else:
            avg_context = X.mean(dim=1)           
            max_context, _ = X.max(dim=1)           
            context_cat = torch.cat([avg_context, max_context], dim=-1) 
        prototype_offsets = self.context_net(context_cat).view(B, self.num_hyperedges, D)  
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets           
        
        X_proj = self.pre_head_proj(X) 
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)
        
        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling 
        logits = logits.view(B, self.num_heads, N, self.num_hyperedges).mean(dim=1) 
        
        logits = self.dropout(logits)  

        return F.softmax(logits, dim=1)

class AdaHGConv(nn.Module):
    """
    Performs the adaptive hypergraph convolution.

    This module contains the two-stage message passing process of hypergraph convolution:
    1. Generates an adaptive participation matrix using AdaHyperedgeGen.
    2. Aggregates vertex features into hyperedge features (vertex-to-edge).
    3. Disseminates hyperedge features back to update vertex features (edge-to-vertex).
    A residual connection is added to the final output.

    Attributes:
        embed_dim (int): The feature dimension of the vertices.
        num_hyperedges (int, optional): The number of hyperedges for the internal generator. Defaults to 16.
        num_heads (int, optional): The number of attention heads for the internal generator. Defaults to 4.
        dropout (float, optional): The dropout rate for the internal generator. Defaults to 0.1.
        context (str, optional): The context type for the internal generator. Defaults to "both".

    Methods:
        forward: Performs the adaptive hypergraph convolution on a batch of vertex features.

    Examples:
        >>> import torch
        >>> model = AdaHGConv(embed_dim=128, num_hyperedges=16, num_heads=8)
        >>> x = torch.randn(2, 256, 128) # (Batch, Num_Nodes, Dim)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 256, 128])
    """
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.edge_generator = AdaHyperedgeGen(embed_dim, num_hyperedges, num_heads, dropout, context)
        self.edge_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )
        self.node_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )
        
    def forward(self, X):
        A = self.edge_generator(X)  
        
        He = torch.bmm(A.transpose(1, 2), X) 
        He = self.edge_proj(He)
        
        X_new = torch.bmm(A, He)  
        X_new = self.node_proj(X_new)
        
        return X_new + X
        
class AdaHGComputation(nn.Module):
    """
    A wrapper module for applying adaptive hypergraph convolution to 4D feature maps.

    This class makes the hypergraph convolution compatible with standard CNN architectures. It flattens a
    4D input tensor (B, C, H, W) into a sequence of vertices (tokens), applies the AdaHGConv layer to
    model high-order correlations, and then reshapes the output back into a 4D tensor.

    Attributes:
        embed_dim (int): The feature dimension of the vertices (equivalent to input channels C).
        num_hyperedges (int, optional): The number of hyperedges for the underlying AdaHGConv. Defaults to 16.
        num_heads (int, optional): The number of attention heads for the underlying AdaHGConv. Defaults to 8.
        dropout (float, optional): The dropout rate for the underlying AdaHGConv. Defaults to 0.1.
        context (str, optional): The context type for the underlying AdaHGConv. Defaults to "both".

    Methods:
        forward: Processes a 4D feature map through the adaptive hypergraph computation layer.

    Examples:
        >>> import torch
        >>> model = AdaHGComputation(embed_dim=64, num_hyperedges=8, num_heads=4)
        >>> x = torch.randn(2, 64, 32, 32) # (B, C, H, W)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=8, dropout=0.1, context="both"):
        super().__init__()
        self.embed_dim = embed_dim
        self.hgnn = AdaHGConv(
            embed_dim=embed_dim,
            num_hyperedges=num_hyperedges,
            num_heads=num_heads,
            dropout=dropout,
            context=context
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2) 
        tokens = self.hgnn(tokens) 
        x_out = tokens.transpose(1, 2).view(B, C, H, W)
        return x_out 

class C3AH(nn.Module):
    """
    A CSP-style block integrating Adaptive Hypergraph Computation (C3AH).

    The input feature map is split into two paths.
    One path is processed by the AdaHGComputation module to model high-order correlations, while the other
    serves as a shortcut. The outputs are then concatenated to fuse features.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        e (float, optional): Expansion ratio for the hidden channels. Defaults to 1.0.
        num_hyperedges (int, optional): The number of hyperedges for the internal AdaHGComputation. Defaults to 8.
        context (str, optional): The context type for the internal AdaHGComputation. Defaults to "both".

    Methods:
        forward: Performs a forward pass through the C3AH module.

    Examples:
        >>> import torch
        >>> model = C3AH(c1=64, c2=128, num_hyperedges=8)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 32, 32])
    """
    def __init__(self, c1, c2, e=1.0, num_hyperedges=8, context="both"):
        super().__init__()
        c_ = int(c2 * e)  
        assert c_ % 16 == 0, "Dimension of AdaHGComputation should be a multiple of 16."
        num_heads = c_ // 16
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = AdaHGComputation(embed_dim=c_, 
                          num_hyperedges=num_hyperedges, 
                          num_heads=num_heads,
                          dropout=0.1,
                          context=context)
        self.cv3 = Conv(2 * c_, c2, 1)  
        
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class FuseModule(nn.Module):
    def __init__(self, c_in, channel_adjust=False):
        super().__init__()

        # Correct downsampling
        self.down = Conv(c_in//2, c_in, k=3, s=2, p=1)

        # Upsampling
        self.reduce = Conv(2* c_in, c_in, k=1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.fuse = Conv(3 * c_in, c_in, 1)

    def forward(self, x):
        x0, x1, x2 = x

        x0 = self.down(x0)   # (75,120) → (38,60)
   
        x2 = self.up(self.reduce (x2))    # (19,30) → (38,60)

        out = torch.cat([x0, x1, x2], dim=1)
        out = self.fuse(out)
        return out

class HyperACE(nn.Module):
    """
    Hypergraph-based Adaptive Correlation Enhancement (HyperACE).

    This is the core module of YOLOv13, designed to model both global high-order correlations and
    local low-order correlations. It first fuses multi-scale features, then processes them through parallel
    branches: two C3AH branches for high-order modeling and a lightweight DSConv-based branch for
    low-order feature extraction.

    Attributes:
        c1 (int): Number of input channels for the fuse module.
        c2 (int): Number of output channels for the entire block.
        n (int, optional): Number of blocks in the low-order branch. Defaults to 1.
        num_hyperedges (int, optional): Number of hyperedges for the C3AH branches. Defaults to 8.
        dsc3k (bool, optional): If True, use DSC3k in the low-order branch; otherwise, use DSBottleneck. Defaults to True.
        shortcut (bool, optional): Whether to use shortcuts in the low-order branch. Defaults to False.
        e1 (float, optional): Expansion ratio for the main hidden channels. Defaults to 0.5.
        e2 (float, optional): Expansion ratio within the C3AH branches. Defaults to 1.
        context (str, optional): Context type for C3AH branches. Defaults to "both".
        channel_adjust (bool, optional): Passed to FuseModule for channel configuration. Defaults to True.

    Methods:
        forward: Performs a forward pass through the HyperACE module.

    Examples:
        >>> import torch
        >>> model = HyperACE(c1=64, c2=256, n=1, num_hyperedges=8)
        >>> x_list = [torch.randn(2, 64, 64, 64), torch.randn(2, 64, 32, 32), torch.randn(2, 64, 16, 16)]
        >>> output = model(x_list)
        >>> print(output.shape)
        torch.Size([2, 256, 32, 32])
    """
    def __init__(self, c1, c2, n=1, num_hyperedges=8, c3k=False, shortcut=False, e1=0.5, e2=1, context="both", channel_adjust=False):
        super().__init__()
        self.c = int(c2 * e1) 
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1) 
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2) if c3k else Bottleneck(self.c, self.c, shortcut=shortcut) for _ in range(n)
        )
        self.fuse = FuseModule(c1, channel_adjust)
        self.branch1 = C3AH(self.c, self.c, e2, num_hyperedges, context)
        self.branch2 = C3AH(self.c, self.c, e2, num_hyperedges, context)
        # 1. Reduce channels half
        self.reduce_c = Conv(c2, c2 // 2, 1)
        # 2. Upsample h2 → h1 : upscale 38×60 → 75×120
        self.up = nn.Upsample(size=(75, 120), mode='nearest')
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.channel_adjust = Conv(c2, c2 * 2, 1)
                    
    def forward(self, X):
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(3, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = out1
        y.append(out2)
        h2 = self.cv2(torch.cat(y, 1))
        h1 =  F.interpolate(x5, size=x4.shape[2:4], mode='nearest') 
        h3 =self.channel_adjust(self.downsample(h2))
        return [h1, h2, h3]
        

class FullPAD_Tunnel(nn.Module):
    """
    A gated fusion module for the Full-Pipeline Aggregation-and-Distribution (FullPAD) paradigm.

    This module implements a gated residual connection used to fuse features. It takes two inputs: the original
    feature map and a correlation-enhanced feature map. It then computes `output = original + gate * enhanced`,
    where `gate` is a learnable scalar parameter that adaptively balances the contribution of the enhanced features.

    Methods:
        forward: Performs the gated fusion of two input feature maps.

    Examples:
        >>> import torch
        >>> model = FullPAD_Tunnel()
        >>> original_feature = torch.randn(2, 64, 32, 32)
        >>> enhanced_feature = torch.randn(2, 64, 32, 32)
        >>> output = model([original_feature, enhanced_feature])
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):
        out = x[0] + self.gate * x[1]
        return out

class Concat(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = Conv(channels, channels//2, 1)

    def forward(self, x, x1):
        return torch.cat([self.conv(x), self.conv(x1)], dim=1) 


# =====================================================
# Spatial Neighborhood Attention
# =====================================================
import torch
import torch.nn as nn
from natten.functional import na2d

class NeighborhoodCrossAttention(nn.Module):
    """
    Cross Neighborhood Attention

    x  : Query feature (current frame)
    x1 : Key/Value feature (previous frame)

    """

    def __init__(
        self,
        dim,
        num_heads=8,
        kernel_size=(3,5),
        dilation=1,
    ):
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        #self.key_dim = int(self.head_dim * attn_ratio)
        #assert self.key_dim > 0, "attn_ratio is too small."

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.q = Conv(dim,self.head_dim * num_heads,1,act=False,)
        self.k = Conv(dim,self.head_dim  * num_heads,1,act=False,)
        self.v = Conv(dim, self.head_dim * num_heads, 1, act=False,)
        # Output projection
        self.proj = Conv(dim,dim,1,act=False,)

    def forward(self, x, x1):

        B, C, H, W = x.shape

        # -----------------------------
        # Linear projections
        # -----------------------------
        q = self.q(x)
        k = self.k(x1)
        v = self.v(x1)

        # -----------------------------
        # Reshape (B,C,H,W) -> (B,H,W,heads,dim)
        # -----------------------------
        q = q.reshape(B,self.num_heads,self.head_dim,H,W,).permute(0, 3, 4, 1, 2)
        k = k.reshape(B,self.num_heads,self.head_dim,H,W,).permute(0, 3, 4, 1, 2)
        v = v.reshape(B,self.num_heads,self.head_dim,H,W,).permute(0, 3, 4, 1, 2)

        # -----------------------------
        # Neighborhood Attention
        # -----------------------------
        out = na2d(
                        q,
                        k,
                        v,
                        kernel_size= self.kernel_size,
                        dilation=self.dilation,
                        backend="flex-fna",
                        run_persistent_kernel=True,
                   )

        # -----------------------------
        # Restore feature map
        # -----------------------------
        out = (out.permute(0, 3, 4, 1, 2).reshape(B, C, H, W))

        # Output projection
        out = self.proj(out)

        return out

# =====================================================
# MLP
# =====================================================
class SCABlock(nn.Module):
    """SCABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    """

    def __init__(self, dim: int, num_heads: int = 4, kernel_size: tuple = (3,5) , shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = NeighborhoodCrossAttention(dim=dim, num_heads=num_heads,kernel_size=kernel_size)
        self.ffn = nn.Sequential(Conv(dim, dim * 2, 1), Conv(dim * 2, dim, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor, xt:torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x,xt) if self.add else self.attn(x,xt)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
        
def gram_matrix(x):   
    batch, channel, h_, w_= x.shape
    before_gram = x.clone()
    before_gram_ = before_gram.view(batch, channel, h_*w_)
    gram = torch.bmm(before_gram_.permute(0, 2, 1), before_gram_)/(channel)
    return gram

class Implicit_S(nn.Module):                
    def __init__(self, h, w,num_imp=4, gram_mode='gram',scale='small'):
        super(Implicit_S, self).__init__()
        self.num_imp = num_imp
        self.gram_mode = gram_mode
        self.h1 = h
        self.w1 = w
        self.h_gram = round(600/64)  # feature
        self.w_gram = round(960/64)
        
        self.scale=scale

        self.imp = nn.Parameter(torch.zeros(1, self.h1*self.w1, self.num_imp))      

        if self.scale=='large':
            self.dynamic_layer = nn.Linear((self.h_gram*self.w_gram)**2, 4*self.num_imp) 
            self.dynamic_layer1 = nn.Linear(4*self.num_imp, self.num_imp)  
        else:
        #  if model =='yolox-s' or 'yolox-m'
            self.dynamic_layer = nn.Linear((self.h_gram*self.w_gram)**2, self.num_imp) 


    def forward(self, gram_):  
        if self.scale=='large':                                            
            parameters1 = F.relu(self.dynamic_layer(gram_))                  
            parameters = self.dynamic_layer1(parameters1).unsqueeze(2)   
        else:    
            parameters = self.dynamic_layer(gram_).unsqueeze(2)  

        parameters_=parameters/parameters.sum(axis=1,keepdims=True)                              # (B,4,1)/(B,1,1)
        
        imp1 = torch.matmul(self.imp, parameters_)                      
        select_imp = imp1.unsqueeze(3).view(-1,1,self.h1,self.w1)        
        return select_imp


    
class ImplicitAdd_S(nn.Module):                    # add
    def __init__(self, h,w, num_imp=4, gram_mode='gram'):
        super().__init__()
        self.h_ = h
        self.w_= w
        self.num_imp = num_imp
        self.gram_mode = gram_mode
        self.imp_add = Implicit_S(self.h_, self.w_, num_imp=self.num_imp, gram_mode=self.gram_mode)
        nn.init.normal_(self.imp_add.imp, std=.02)

    def forward(self, x):
        h_current,w_current = x.shape[2:4]
        imp_add_scale = F.interpolate(self.imp_add(x),size=(h_current,w_current), mode='bilinear')
        x_ = imp_add_scale.expand_as(x) + x
        return x_


class ImplicitMul_S(nn.Module):                    # mul
    def __init__(self, h, w, num_imp=4, gram_mode='gram'):
        super().__init__()
        self.h_ = h
        self.w_ = w
        self.num_imp = num_imp
        self.gram_mode = gram_mode
        self.imp_mul = Implicit_S(self.h_, self.w_, num_imp=self.num_imp, gram_mode=self.gram_mode)
        nn.init.normal_(self.imp_mul.imp, mean=1., std=.02)

    def forward(self, x):
        h_current,w_current = x.shape[2:4]
        imp_mul_scale = F.interpolate(self.imp_mul(x), size=(h_current,w_current), mode='bilinear')
        x_ = imp_mul_scale.expand_as(x) * x
        return x_
class ImplicitMul_S_8(nn.Module):                  # mul  &  Specify the resolution
    def __init__(self, h, w, num_imp=4, gram_mode='gram'):
        super().__init__()
        self.h_ = round(600/16)
        self.w_ = round(960/16)
        self.num_imp = num_imp
        self.gram_mode = gram_mode
        self.imp_mul = Implicit_S(self.h_, self.w_, num_imp=self.num_imp, gram_mode=self.gram_mode)
        nn.init.normal_(self.imp_mul.imp, mean=1., std=.02)

    def forward(self, x):
        x16=F.interpolate(x,size=(self.h_,self.w_),mode='bilinear')
        h_current,w_current = x.shape[2:4]
        imp_mul_scale = F.interpolate(self.imp_mul(x16), size=(h_current,w_current), mode='bilinear')
        x_ = imp_mul_scale.expand_as(x) * x
        return x_


class ImplicitShift_S(nn.Module):                    # shift                      
    def __init__(self, h, w, num_imp=4, gram_mode='gram',scale='large'):
        super().__init__()
        self.h_ = h
        self.w_ = w    
        self.h_gram = round(600/64)  # feature
        self.w_gram = round(960/64)  
        self.num_imp = num_imp
        self.gram_mode = gram_mode

        self.scale=scale

        self.imp_add = Implicit_S(self.h_, self.w_, num_imp=self.num_imp, gram_mode=self.gram_mode,scale=self.scale)
        nn.init.normal_(self.imp_add.imp,mean=0., std=.02)
        self.imp_mul = Implicit_S(self.h_, self.w_, num_imp=self.num_imp, gram_mode=self.gram_mode,scale=self.scale)
        nn.init.normal_(self.imp_mul.imp, mean=1., std=.02)

    def forward(self, x):
        h_current,w_current = x.shape[2:4]
        x64=F.interpolate(x,size=(self.h_gram,self.w_gram),mode='bilinear')
        gram_scale = gram_matrix(x64).unsqueeze(1).squeeze(1)            # (B ,h*w,h*w)--->(B ,1, h*w,h*w)
        gram_ = gram_scale.view(-1, (self.h_gram*self.w_gram)**2)            # (B,h*w*h*w)   

        imp_mul_scale = F.interpolate(self.imp_mul(gram_), size=(h_current,w_current), mode='bilinear')
        imp_add_scale = F.interpolate(self.imp_add(gram_),size=(h_current,w_current), mode='bilinear')

        x_ = imp_mul_scale.expand_as(x) * x +imp_add_scale.expand_as(x)  
        return x_

    
class ImplicitShift_S_8(nn.Module):          # shift  & Specify the resolution                                 
    def __init__(self, h, w,num_imp=4, gram_mode='gram',scale='large'):
        super().__init__()
        self.h_ = round(600/16)  #  implicit knowledge 
        self.w_ = round(960/16)
        
        self.h_gram = round(600/64)  # gram feature scale
        self.w_gram = round(960/64)
        
        self.num_imp = num_imp
        self.gram_mode = gram_mode
        self.scale=scale

        self.imp_add = Implicit_S(self.h_, self.w_, num_imp=self.num_imp, gram_mode=self.gram_mode,scale=self.scale)
        nn.init.normal_(self.imp_add.imp,mean=0., std=.02)
        self.imp_mul = Implicit_S(self.h_, self.w_, num_imp=self.num_imp, gram_mode=self.gram_mode,scale=self.scale)
        nn.init.normal_(self.imp_mul.imp, mean=1., std=.02)

    def forward(self, x):
        h_current,w_current = x.shape[2:4]
        x64=F.interpolate(x,size=(self.h_gram,self.w_gram),mode='bilinear')
        gram_scale = gram_matrix(x64).unsqueeze(1).squeeze(1)          
        gram_ = gram_scale.view(-1, (self.h_gram*self.w_gram)**2)        

        imp_mul_scale = F.interpolate(self.imp_mul(gram_), size=(h_current,w_current), mode='bilinear')
        imp_add_scale = F.interpolate(self.imp_add(gram_),size=(h_current,w_current), mode='bilinear')

        x_ = imp_mul_scale.expand_as(x) * x +imp_add_scale.expand_as(x) 
        return x_

class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)

class Detect(nn.Module):
    """YOLO Detect head for object detection models.

    This class implements the detection head used in YOLO models for predicting bounding boxes and class probabilities.
    It supports both training and inference modes, with optional end-to-end detection capabilities.

    Attributes:
        dynamic (bool): Force grid reconstruction.
        export (bool): Export mode flag.
        format (str): Export format.
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum detections per image.
        shape (tuple): Input shape.
        anchors (torch.Tensor): Anchor points.
        strides (torch.Tensor): Feature map strides.
        legacy (bool): Backward compatibility for v3/v5/v8/v9 models.
        xyxy (bool): Output format, xyxy or xywh.
        nc (int): Number of classes.
        nl (int): Number of detection layers.
        reg_max (int): DFL channels.
        no (int): Number of outputs per anchor.
        stride (torch.Tensor): Strides computed during build.
        cv2 (nn.ModuleList): Convolution layers for box regression.
        cv3 (nn.ModuleList): Convolution layers for classification.
        dfl (nn.Module): Distribution Focal Loss layer.
        one2one_cv2 (nn.ModuleList): One-to-one convolution layers for box regression.
        one2one_cv3 (nn.ModuleList): One-to-one convolution layers for classification.

    Methods:
        forward: Perform forward pass and return predictions.
        forward_end2end: Perform forward pass for end-to-end detection.
        bias_init: Initialize detection head biases.
        decode_bboxes: Decode bounding boxes from predictions.
        postprocess: Post-process model predictions.

    Examples:
        Create a detection head for 80 classes
        >>> detect = Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = detect(x)
    """

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output

    def __init__(self, nc: int = 8, reg_max=16, end2end=False, ch: tuple = (256, 512, 1024)):
        """Initialize the YOLO detection layer with specified number of classes and channels.

        Args:
            nc (int): Number of classes.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        #self.stride = torch.zeros(self.nl)  # strides computed during build
        self.stride = torch.tensor([8.,16.,32.])
        #c1 = ch[0]
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], 80, min(self.nc, 10))  # channels
        
        self.cv2 = nn.ModuleList(
            #nn.Sequential(
            #nn.Sequential(CNN(x, c1, 3,groups= math.gcd(x,c1) ), CNN(c1, c2, 1)),
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
                                 for x in ch
        )
        self.cv3 = (nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v5/v5/v8/v9/11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    @property
    def end2end(self):
        """Checks if the model has one2one for v5/v5/v8/v9/11 backward compatibility."""
        return getattr(self, "_end2end", True) and hasattr(self, "one2one")

    @end2end.setter
    def end2end(self, value):
        """Override the end-to-end detection mode."""
        self._end2end = value

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (dict[str, torch.Tensor]): List of feature maps from different detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get decoded boxes based on anchors and strides."""
        shape = x["feats"][0].shape  # BCHW
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape

        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):  # from
            s = float(self.stride[i])
            #a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (round(600 / s) * round(960 / s))
            )  # cls (.01 objects, 8 classes, 600*960 img)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):  # from
                #a[-1].bias.data[:] = 2.0  # box
                b[-1].bias.data[: self.nc] = math.log(
                    5 / self.nc / (round(600 / s) * round(960 / s))
                )  # cls (.01 objects, 8 classes, 600*960 img)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        boxes, scores = preds.split([4, self.nc], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    def get_topk_index(self, scores: torch.Tensor, max_det: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get top-k indices from scores.

        Args:
            scores (torch.Tensor): Scores tensor with shape (batch_size, num_anchors, num_classes).
            max_det (int): Maximum detections per image.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): Top scores, class indices, and filtered indices.
        """
        batch_size, anchors, nc = scores.shape  # i.e. shape(16,8400,84)
        # Use max_det directly during export for TensorRT compatibility (requires k to be constant),
        # otherwise use min(max_det, anchors) for safety with small inputs during Python inference
        k = max_det if self.export else min(max_det, anchors)
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size)[..., None], index // nc]  # original index
        return scores[..., None], (index % nc)[..., None].float(), idx

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = None




class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of pooling iterations.
            shortcut (bool): Whether to use shortcut connection.

        Notes:
            With the default k=5 and n=3, this module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple = (3, 3), e: float = 0.5):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes of cv1 and cv2, each an int or an (h, w) tuple.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
        format (str | None): Export format set at export time; "coreml" uses `F.scaled_dot_product_attention`.
    """

    format = None

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        if self.format == "coreml" and hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1))
            x = x.transpose(-2, -1).reshape(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        else:
            attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
            x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C3k2(C2f):
    """C2f variant whose inner blocks are Bottleneck, C3k, or Bottleneck + PSABlock attention modules."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            attn (bool): Whether to use attention blocks.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

