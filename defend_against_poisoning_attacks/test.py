import torch

# 加载 .pt 文件
file_path = 'D:/Test/results/Revision_1/avg/fmnist_45_450_0.1_0.1_avg_MR_09202122.pt'
# file_path = 'D:/Test/results/Revision_1/jzx_test/cifar10_20_100_0.1_0.03_jzx_test_MR_09201536.pt'

# file_path = 'D:/Test/results/Revision_1/jzx_no_defense/cifar10_20_400_0.1_0.05_jzx_no_defense_MR_09161018.pt'

# file_path = 'D:/Test/results/Revision_1/flame/cifar10_20_225_0.1_0.1_flame_MR_09211500.pt'

model_data = torch.load(file_path)

# if isinstance(model_data, dict):
#     print("Keys:", model_data.keys())

print(f"accuracy: {model_data['accuracy']}")
print(f"poison_accuracy: {model_data['poison_accuracy']}")

print(f"ASR: {model_data['ASR']}")
print(f"TER: {model_data['TER']}")

print(f"MAX-ASR: {model_data['MAX-ASR']}")