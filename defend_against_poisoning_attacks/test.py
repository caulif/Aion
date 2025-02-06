import torch

file_path = 'D:/Test/results/Revision_1/avg/fmnist_45_450_0.1_0.1_avg_MR_09202122.pt'


model_data = torch.load(file_path)



print(f"accuracy: {model_data['accuracy']}")
print(f"poison_accuracy: {model_data['poison_accuracy']}")

print(f"ASR: {model_data['ASR']}")
print(f"TER: {model_data['TER']}")

print(f"MAX-ASR: {model_data['MAX-ASR']}")