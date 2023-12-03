import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t) #pair函数：这是一个辅助函数，用于从一个数值或元组中提取两个值。它将确保给定的值被分配给两个不同的变量，通常用于处理图像尺寸和patch尺寸的输入。

# classes

class FeedForward(nn.Module): #FeedForward类：这是一个前馈神经网络模块，用于对输入进行线性变换、GELU激活函数和dropout操作。它通常在Transformer的每个层中用于处理特征。
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module): #这是注意力机制模块，用于计算注意力分布。它包括对输入进行Layer Normalization、计算注意力分布、应用dropout和输出处理。注意力机制通常用于捕获输入序列中不同位置的依赖关系。
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim) #project_out：一个布尔值，指示是否需要将注意力输出映射回原始维度

        self.heads = heads
        self.scale = dim_head ** -0.5 #理论部分的除权数，通过缩放因子，确保注意力分布的尺度适当

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1) #dim=-1意味着对最后一个维度进行softmax计算
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity() #nn.Identity() 不对输入进行任何变换，只是将输入复制到输出。

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1) # 通过 chunk(3, dim=-1) 将 qkv 分成三部分 x = torch.randn(2, 10) chunks = x.chunk(2, dim=1)  # 在维度1上分割成2个块 # chunks[0] 包含 x 的前5列，chunks[1] 包含 x 的后5列
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) #map 是一个Python内置函数，它的主要作用是将一个函数应用于可迭代对象（如列表、元组等），并返回一个包含函数应用结果的新可迭代对象。

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size) #pair 函数用于从一个给定的数值或元组中提取两个值，并将它们分配给两个不同的变量。在上面的代码片段中，pair 函数被用来从图像尺寸和图像patch尺寸中提取高度和宽度。
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size' #确保尺寸能被整除

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size), #'b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)' 是 Rearrange 函数的格式字符串，用于指定如何重新排列输入数据的维度。这个字符串描述了从输入张量到输出张量的映射方式。 Rearrange部分有不会的可以参考收藏夹里的CSDN文章
            #所以实际上这部分代码只是利用维度重排序以将3D的数据适用在2D结构的VIT上
            nn.LayerNorm(patch_dim), #patch_dim就是重排序之后的最后一个维度
            nn.Linear(patch_dim, dim), #示将输入的 patch_dim 维度特征线性映射到 dim 维度特征
            nn.LayerNorm(dim), #LayerNorm层具体操作：使用计算得到的均值和标准差来标准化每个样本，即每个样本减去均值，除以方差
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) #nn.Parameter 允许将张量标记为模型的参数，使其能够参与训练，并在反向传播时进行更新。这是构建深度学习模型和训练它们的重要组成部分。self.pos_embedding 是一个可学习的参数，用于捕捉位置信息。它的形状是 (1, num_patches + 1, dim)，其中 num_patches 表示输入图像被划分成的图像块的数量，加 1 是为了额外的位置嵌入（通常用于类别标记）。这些位置嵌入将被加到输入嵌入中，以帮助模型理解每个图像块的位置信息。
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) #self.cls_token 也是一个可学习的参数，用于表示一个额外的类别标记。它的形状是 (1, 1, dim)，其中 dim 是Transformer的输入和输出维度。这个类别标记通常与图像块的表示连接在一起，以捕获全局信息。
        self.dropout = nn.Dropout(emb_dropout) #self.dropout 是一个Dropout层，用于在输入嵌入上应用丢弃操作。丢弃操作有助于防止过拟合，它以概率 emb_dropout 随机将输入中的一些元素设置为零。

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
