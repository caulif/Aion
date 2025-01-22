import numpy as np
import torch


class Tool:
    """
    工具类
    用于处理 NIID的网络/模型 与 flamingo的向量 之间的转换
    注意输入的net是全局模型的局部，而不是完整的全局模型
    """

    @staticmethod
    # 获取网络参数长度
    def net_len(net) -> int:
        length = 0
        for param_tensor in net.cpu().state_dict():
            length += np.prod(net.cpu().state_dict()[param_tensor].shape)
        return length

    @staticmethod
    # 获取网络参数规格
    def net_shape(net) -> list:
        shape = []
        for param_tensor in net.cpu().state_dict():
            shape.append(net.cpu().state_dict()[param_tensor].shape)
        return shape

    @staticmethod
    # NIID网络参数 -> flamingo向量
    def net2vec(net) -> np.ndarray:
        vec = []
        for param_tensor in net.cpu().state_dict():
            vec.extend(net.cpu().state_dict()[param_tensor].reshape(-1))
        return np.array(vec)
    @staticmethod
    # flamingo向量 -> NIID网络参数
    def vec2net(vec, net):
        start = 0
        state_dict = net.state_dict()  # 获取参数副本
        for param_tensor in state_dict:
            end = start + np.prod(state_dict[param_tensor].shape)
            state_dict[param_tensor] = torch.tensor(
                vec[start:end].reshape(state_dict[param_tensor].shape))
            start = end
        net.load_state_dict(state_dict)  # 加载更新后的参数
        return net
