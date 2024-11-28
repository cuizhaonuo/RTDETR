'''by lyuwenyu
'''

import torch 
import torch.nn as nn



class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FrozenBatchNorm2d(nn.Module):
    """copy and modified from https://github.com/facebookresearch/detr/blob/master/models/backbone.py
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.

    当你调用 model.load_state_dict(state_dict) 时，PyTorch 内部会自动调用 _load_from_state_dict() 方法，流程如下：
    1. PyTorch 会创建 missing_keys、unexpected_keys 和 error_msgs 列表。
    2. 为每个子模块递归调用 _load_from_state_dict() 方法，并传入对应的 prefix。
    3. 在子模块中（例如 FrozenBatchNorm2d），会根据需要删除无用的键（如 num_batches_tracked）。
    4. 调用父类的 _load_from_state_dict() 方法，加载参数和缓冲区。
    """
    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        n = num_features
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps
        self.num_features = n 

    # 在加载预训练模型时，running_mean 和 running_var 会从预训练模型的权重文件中加载
    # 子类重写父类的方法，并保存父类方法的功能
    # _load_from_state_dict 是父类 nn.Module 的私有方法，当 load_state_dict 方法调用时，该方法自动调用
    # num_batches_tracked 是 BatchNorm 层的一个缓冲区，用于跟踪处理的批次数
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 在加载预训练模型的权重时，预训练模型通常是标准的 BatchNorm2d，其中包含 num_batches_tracked。
        # 如果不删除这个键，调用 load_state_dict() 时，PyTorch 会尝试将 num_batches_tracked 加载到 FrozenBatchNorm2d 中，
        # 但这个类没有这个缓冲区，导致键不匹配错误。
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        # rsqrt() 计算 tensor 每个元素的平方根的倒数 || reciprocal square root
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        # 当 w = 1, b = 0 时，该公式即常见的标准化公式
        return x * scale + bias

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}".format(**self.__dict__)
        )


def get_activation(act: str, inpace: bool=True):
    '''get activation
    '''
    act = act.lower()
    
    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()
        
    elif act is None:
        # nn.Identity() 是一个非常简单的层，它不对输入数据做任何修改，只是将输入直接输出。
        m = nn.Identity()
    
    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m 
