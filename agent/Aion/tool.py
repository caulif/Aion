import numpy as np
import torch


class Tool:
    """
    Utility class
    Used to handle the conversion between NIID network/model and flamingo vectors.
    Note that the input 'net' is a part of the global model, not the complete global model.
    """

    @staticmethod
    # Get the length of network parameters
    def net_len(net) -> int:
        length = 0
        for param_tensor in net.cpu().state_dict():
            length += np.prod(net.cpu().state_dict()[param_tensor].shape)
        return length

    @staticmethod
    # Get the shapes of network parameters
    def net_shape(net) -> list:
        shape = []
        for param_tensor in net.cpu().state_dict():
            shape.append(net.cpu().state_dict()[param_tensor].shape)
        return shape

    @staticmethod
    # Convert NIID network parameters to a flamingo vector
    def net2vec(net) -> np.ndarray:
        vec = []
        for param_tensor in net.cpu().state_dict():
            vec.extend(net.cpu().state_dict()[param_tensor].reshape(-1))
        return np.array(vec)
    @staticmethod
    # Convert a flamingo vector to NIID network parameters
    def vec2net(vec, net):
        start = 0
        state_dict = net.state_dict()  # Get a copy of the parameters
        for param_tensor in state_dict:
            end = start + np.prod(state_dict[param_tensor].shape)
            state_dict[param_tensor] = torch.tensor(
                vec[start:end].reshape(state_dict[param_tensor].shape))
            start = end
        net.load_state_dict(state_dict)  # Load the updated parameters
        return net