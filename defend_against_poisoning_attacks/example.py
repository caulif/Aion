import torch

tensor_1 = torch.tensor([1.05]*4)
tensor_2 = torch.tensor([1.1]*4)
tensor_3 = torch.tensor([-1.15]*4)
tensor_4 = torch.tensor([1.2]*4)

stacked_tensors = torch.stack([tensor_1, tensor_2, tensor_3, tensor_4], dim=0)

tensor_mean = torch.mean(stacked_tensors, dim=0)

# Calculate the L2 norm (Euclidean norm) for each tensor and the column mean
l2_norm_tensor_1 = torch.norm(tensor_1, p=2)
l2_norm_tensor_2 = torch.norm(tensor_2, p=2)
l2_norm_tensor_3 = torch.norm(tensor_3, p=2)
l2_norm_tensor_4 = torch.norm(tensor_4, p=2)
l2_norm_column_mean = torch.norm(tensor_mean, p=2)

print(l2_norm_tensor_1)

print(l2_norm_tensor_2)

print(l2_norm_tensor_3)

print(l2_norm_tensor_4)

print(l2_norm_column_mean)