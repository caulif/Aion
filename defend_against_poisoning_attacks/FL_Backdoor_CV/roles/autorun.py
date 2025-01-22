import subprocess

# # 定义weights的值列表
# weights = [0.1, 0.2, 0.3, 0.4, 0.5]

# common_command = [
#     "python", "./FL_Backdoor_CV/roles/trainer.py",
#     "--dataset=cifar10",
#     "--params=utils/cifar10_params.yaml",
#     "--class_imbalance=0",
#     "--balance=0.99",
#     "--classes_per_client=2",
#     "--resume=1",
#     "--resumed_name=jzx_test_MR_cifar10/jzx_test_1000.pth",
#     "--rounds=1000",
#     "--participant_population=500",
#     "--participant_sample_size=100",
#     "--number_of_adversaries=20",
#     "--mal_boost=400",
#     "--is_poison=1",
#     "--aggregation_rule=jzx_test",
#     "--random_compromise=0",
#     "--retrain_rounds=60",
#     "--poison_prob=0.5",
#     "--device=cuda:0",
#     "--local_lr=0.1",
#     "--local_lr_decay=0.991",
#     "--decay_step=5",
#     "--local_lr_min=0.001",
#     "--global_lr=1",
#     "--global_lr_decay=1",
#     "--batch_size=64",
#     "--momentum=0.9",
#     "--decay=0.0005",
#     "--attack_mode=MR",
#     "--num_poisoned_samples=6",
#     "--gradmask_ratio=0.5",
#     "--multi_objective_num=4",
#     "--alternating_minimization=0",
#     "--record_step=100",
#     "--record_res_step=10",
#     "--min_threshold=0.1"
# ]

# # 遍历weights的值，并执行每个命令
# for weight in weights:
#     command = common_command + [f"--weight={weight}"]
#     print(f"Executing: {' '.join(command)}")
#     subprocess.run(command, check=True)


# common_command = [
#     "python", "./FL_Backdoor_CV/roles/trainer.py",
#     "--dataset=cifar10",
#     "--params=utils/cifar10_params.yaml",
#     "--class_imbalance=0",
#     "--balance=0.99",
#     "--classes_per_client=2",
#     "--resume=1",
#     "--resumed_name=jzx_test_MR_cifar10/jzx_test_1000.pth",
#     "--rounds=1000",
#     "--participant_population=500",
#     "--participant_sample_size=100",
#     "--number_of_adversaries=10",
#     "--mal_boost=200",
#     "--is_poison=1",
#     "--aggregation_rule=jzx_test",
#     "--random_compromise=0",
#     "--retrain_rounds=60",
#     "--poison_prob=0.5",
#     "--device=cuda:0",
#     "--local_lr=0.1",
#     "--local_lr_decay=0.991",
#     "--decay_step=5",
#     "--local_lr_min=0.001",
#     "--global_lr=1",
#     "--global_lr_decay=1",
#     "--batch_size=64",
#     "--momentum=0.9",
#     "--decay=0.0005",
#     "--attack_mode=MR",
#     "--num_poisoned_samples=6",
#     "--gradmask_ratio=0.5",
#     "--multi_objective_num=4",
#     "--alternating_minimization=0",
#     "--record_step=100",
#     "--record_res_step=10",
#     "--min_threshold=0.1"
# ]

# # 遍历weights的值，并执行每个命令
# for weight in weights:
#     command = common_command + [f"--weight={weight}"]
#     print(f"Executing: {' '.join(command)}")
#     subprocess.run(command, check=True)




# jzx_no_defense
# common_command = [
#     "python", "./FL_Backdoor_CV/roles/trainer.py",
#     "--dataset=cifar10",
#     "--params=utils/cifar10_params.yaml",
#     "--class_imbalance=0",
#     "--balance=0.99",
#     "--classes_per_client=2",
#     "--resume=1",
#     "--resumed_name=jzx_test_MR_cifar10/jzx_test_1000.pth",
#     "--rounds=1000",
#     "--participant_population=500",
#     "--participant_sample_size=100",
#     "--number_of_adversaries=10",
#     "--mal_boost=200",
#     "--is_poison=1",
#     "--aggregation_rule=jzx_no_defense",
#     "--random_compromise=0",
#     "--retrain_rounds=60",
#     "--poison_prob=0.5",
#     "--device=cuda:0",
#     "--local_lr=0.1",
#     "--local_lr_decay=0.991",
#     "--decay_step=5",
#     "--local_lr_min=0.001",
#     "--global_lr=1",
#     "--global_lr_decay=1",
#     "--batch_size=64",
#     "--momentum=0.9",
#     "--decay=0.0005",
#     "--attack_mode=MR",
#     "--num_poisoned_samples=6",
#     "--gradmask_ratio=0.5",
#     "--multi_objective_num=4",
#     "--alternating_minimization=0",
#     "--record_step=100",
#     "--record_res_step=10",
#     "--min_threshold=0.1"
# ]

# # 遍历weights的值，并执行每个命令
# for weight in weights:
#     command = common_command + [f"--weight={weight}"]
#     print(f"Executing: {' '.join(command)}")
#     subprocess.run(command, check=True)


# common_command = [
#     "python", "./FL_Backdoor_CV/roles/trainer.py",
#     "--dataset=cifar10",
#     "--params=utils/cifar10_params.yaml",
#     "--class_imbalance=0",
#     "--balance=0.99",
#     "--classes_per_client=2",
#     "--resume=1",
#     "--resumed_name=jzx_test_MR_cifar10/jzx_test_1000.pth",
#     "--rounds=1000",
#     "--participant_population=500",
#     "--participant_sample_size=100",
#     "--number_of_adversaries=20",
#     "--mal_boost=400",
#     "--is_poison=1",
#     "--aggregation_rule=jzx_no_defense",
#     "--random_compromise=0",
#     "--retrain_rounds=60",
#     "--poison_prob=0.5",
#     "--device=cuda:0",
#     "--local_lr=0.1",
#     "--local_lr_decay=0.991",
#     "--decay_step=5",
#     "--local_lr_min=0.001",
#     "--global_lr=1",
#     "--global_lr_decay=1",
#     "--batch_size=64",
#     "--momentum=0.9",
#     "--decay=0.0005",
#     "--attack_mode=MR",
#     "--num_poisoned_samples=6",
#     "--gradmask_ratio=0.5",
#     "--multi_objective_num=4",
#     "--alternating_minimization=0",
#     "--record_step=100",
#     "--record_res_step=10",
#     "--min_threshold=0.1"
# ]

# # 遍历weights的值，并执行每个命令
# for weight in weights:
#     command = common_command + [f"--weight={weight}"]
#     print(f"Executing: {' '.join(command)}")
#     subprocess.run(command, check=True)



# weights = [0.6, 0.7, 0.8, 0.9, 1]

# # FMNIST
# common_command = [
#     "python", "./FL_Backdoor_CV/roles/trainer.py",
#     "--dataset=fmnist",
#     "--params=utils/fmnist_params.yaml",
#     "--class_imbalance=0",
#     "--balance=0.99",
#     "--classes_per_client=2",
#     "--resume=1",
#     "--resumed_name=avg_MR_fmnist/avg_300.pth",
#     "--rounds=1000",
#     "--participant_population=500",
#     "--participant_sample_size=100",
#     "--number_of_adversaries=10",
#     "--mal_boost=100",
#     "--is_poison=1",
#     "--aggregation_rule=jzx_no_defense",
#     "--random_compromise=0",
#     "--retrain_rounds=60",
#     "--poison_prob=0.5",
#     "--device=cuda:0",
#     "--local_lr=0.1",
#     "--local_lr_decay=0.991",
#     "--decay_step=5",
#     "--local_lr_min=0.001",
#     "--global_lr=1",
#     "--global_lr_decay=1",
#     "--batch_size=64",
#     "--momentum=0.9",
#     "--decay=0.0005",
#     "--attack_mode=MR",
#     "--num_poisoned_samples=6",
#     "--gradmask_ratio=0.5",
#     "--multi_objective_num=4",
#     "--alternating_minimization=0",
#     "--record_step=100",
#     "--record_res_step=10",
#     "--min_threshold=0.1"
# ]

# # 遍历weights的值，并执行每个命令
# for weight in weights:
#     command = common_command + [f"--weight={weight}"]
#     print(f"Executing: {' '.join(command)}")
#     subprocess.run(command, check=True)



# weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# common_command = [
#     "python", "./FL_Backdoor_CV/roles/trainer.py",
#     "--dataset=fmnist",
#     "--params=utils/fmnist_params.yaml",
#     "--class_imbalance=0",
#     "--balance=0.99",
#     "--classes_per_client=2",
#     "--resume=1",
#     "--resumed_name=avg_MR_fmnist/avg_300.pth",
#     "--rounds=1000",
#     "--participant_population=500",
#     "--participant_sample_size=100",
#     "--number_of_adversaries=20",
#     "--mal_boost=200",
#     "--is_poison=1",
#     "--aggregation_rule=jzx_no_defense",
#     "--random_compromise=0",
#     "--retrain_rounds=60",
#     "--poison_prob=0.5",
#     "--device=cuda:0",
#     "--local_lr=0.1",
#     "--local_lr_decay=0.991",
#     "--decay_step=5",
#     "--local_lr_min=0.001",
#     "--global_lr=1",
#     "--global_lr_decay=1",
#     "--batch_size=64",
#     "--momentum=0.9",
#     "--decay=0.0005",
#     "--attack_mode=MR",
#     "--num_poisoned_samples=6",
#     "--gradmask_ratio=0.5",
#     "--multi_objective_num=4",
#     "--alternating_minimization=0",
#     "--record_step=100",
#     "--record_res_step=10",
#     "--min_threshold=0.1"
# ]

# # 遍历weights的值，并执行每个命令
# for weight in weights:
#     command = common_command + [f"--weight={weight}"]
#     print(f"Executing: {' '.join(command)}")
#     subprocess.run(command, check=True)





# # 定义number_of_adversaries的值列表
# aggregation_rules=['flame']
# numbers_of_adversaries=[15, 25, 35, 45]

# common_command = [
#     "python", "./FL_Backdoor_CV/roles/trainer.py",
#     "--dataset=cifar10",
#     "--params=utils/cifar10_params.yaml",
#     "--class_imbalance=0",
#     "--balance=0.99",
#     "--classes_per_client=2",
#     "--resume=1",
#     "--resumed_name=jzx_test_MR_cifar10/jzx_test_1000.pth",
#     "--rounds=1000",
#     "--participant_population=500",
#     "--participant_sample_size=100",
#     "--is_poison=1",
#     "--random_compromise=0",
#     "--retrain_rounds=60",
#     "--poison_prob=0.5",
#     "--device=cuda:0",
#     "--local_lr=0.1",
#     "--local_lr_decay=0.991",
#     "--decay_step=5",
#     "--local_lr_min=0.001",
#     "--global_lr=1",
#     "--global_lr_decay=1",
#     "--batch_size=64",
#     "--momentum=0.9",
#     "--decay=0.0005",
#     "--attack_mode=MR",
#     "--num_poisoned_samples=6",
#     "--gradmask_ratio=0.5",
#     "--multi_objective_num=4",
#     "--alternating_minimization=0",
#     "--record_step=100",
#     "--record_res_step=10",
#     "--weight=0.1",
#     "--min_threshold=0.1"
# ]

# # 遍历number_of_adversaries的值，并执行每个命令
# for aggregation_rule in aggregation_rules:
#     for number_of_adversaries in numbers_of_adversaries:
#         command = common_command + [f"--number_of_adversaries={number_of_adversaries}"] \
#             + [f"--mal_boost={number_of_adversaries * 20}"] + [f"--aggregation_rule={aggregation_rule}"]
#         print(f"Executing: {' '.join(command)}")
#         subprocess.run(command, check=True)



# aggregation_rules=['jzx_test', 'flame']
# numbers_of_adversaries=[5, 15, 25, 35, 45]

# # fmnist
# common_command = [
#     "python", "./FL_Backdoor_CV/roles/trainer.py",
#     "--dataset=fmnist",
#     "--params=utils/fmnist_params.yaml",
#     "--class_imbalance=0",
#     "--balance=0.99",
#     "--classes_per_client=2",
#     "--resume=1",
#     "--resumed_name=avg_MR_fmnist/avg_300.pth",
#     "--rounds=1000",
#     "--participant_population=500",
#     "--participant_sample_size=100",
#     "--is_poison=1",
#     "--random_compromise=0",
#     "--retrain_rounds=60",
#     "--poison_prob=0.5",
#     "--device=cuda:0",
#     "--local_lr=0.1",
#     "--local_lr_decay=0.991",
#     "--decay_step=5",
#     "--local_lr_min=0.001",
#     "--global_lr=1",
#     "--global_lr_decay=1",
#     "--batch_size=64",
#     "--momentum=0.9",
#     "--decay=0.0005",
#     "--attack_mode=MR",
#     "--num_poisoned_samples=6",
#     "--gradmask_ratio=0.5",
#     "--multi_objective_num=4",
#     "--alternating_minimization=0",
#     "--record_step=100",
#     "--record_res_step=10",
#     "--weight=0.1",
#     "--min_threshold=0.1",
# ]

# # 遍历number_of_adversaries的值，并执行每个命令
# for aggregation_rule in aggregation_rules:
#     for number_of_adversaries in numbers_of_adversaries:
#         command = common_command + [f"--number_of_adversaries={number_of_adversaries}"] \
#             + [f"--mal_boost={number_of_adversaries * 10}"] + [f"--aggregation_rule={aggregation_rule}"]
#         print(f"Executing: {' '.join(command)}")
#         subprocess.run(command, check=True)



aggregation_rules=['avg']
# numbers_of_adversaries=[35, 25, 15, 5]

# common_command = [
#     "python", "./FL_Backdoor_CV/roles/trainer.py",
#     "--dataset=cifar10",
#     "--params=utils/cifar10_params.yaml",
#     "--class_imbalance=0",
#     "--balance=0.99",
#     "--classes_per_client=2",
#     "--resume=1",
#     "--resumed_name=jzx_test_MR_cifar10/jzx_test_1000.pth",
#     "--rounds=1000",
#     "--participant_population=500",
#     "--participant_sample_size=100",
#     "--is_poison=1",
#     "--random_compromise=0",
#     "--retrain_rounds=60",
#     "--poison_prob=0.5",
#     "--device=cuda:0",
#     "--local_lr=0.1",
#     "--local_lr_decay=0.991",
#     "--decay_step=5",
#     "--local_lr_min=0.001",
#     "--global_lr=1",
#     "--global_lr_decay=1",
#     "--batch_size=64",
#     "--momentum=0.9",
#     "--decay=0.0005",
#     "--attack_mode=MR",
#     "--num_poisoned_samples=6",
#     "--gradmask_ratio=0.5",
#     "--multi_objective_num=4",
#     "--alternating_minimization=0",
#     "--record_step=100",
#     "--record_res_step=10",
#     "--weight=0.1",
#     "--min_threshold=0.1",
#     "--s_norm=5"
# ]

# # 遍历number_of_adversaries的值，并执行每个命令
# for aggregation_rule in aggregation_rules:
#     for number_of_adversaries in numbers_of_adversaries:
#         command = common_command + [f"--number_of_adversaries={number_of_adversaries}"] \
#             + [f"--mal_boost={number_of_adversaries * 20}"] + [f"--aggregation_rule={aggregation_rule}"]
#         print(f"Executing: {' '.join(command)}")
#         subprocess.run(command, check=True)


numbers_of_adversaries=[45, 35, 25, 15, 5]

# fmnist
common_command = [
    "python", "./FL_Backdoor_CV/roles/trainer.py",
    "--dataset=fmnist",
    "--params=utils/fmnist_params.yaml",
    "--class_imbalance=0",
    "--balance=0.99",
    "--classes_per_client=2",
    "--resume=1",
    "--resumed_name=avg_MR_fmnist/avg_300.pth",
    "--rounds=1000",
    "--participant_population=500",
    "--participant_sample_size=100",
    "--is_poison=1",
    "--random_compromise=0",
    "--retrain_rounds=60",
    "--poison_prob=0.5",
    "--device=cuda:0",
    "--local_lr=0.1",
    "--local_lr_decay=0.991",
    "--decay_step=5",
    "--local_lr_min=0.001",
    "--global_lr=1",
    "--global_lr_decay=1",
    "--batch_size=64",
    "--momentum=0.9",
    "--decay=0.0005",
    "--attack_mode=MR",
    "--num_poisoned_samples=6",
    "--gradmask_ratio=0.5",
    "--multi_objective_num=4",
    "--alternating_minimization=0",
    "--record_step=100",
    "--record_res_step=10",
    "--weight=0.1",
    "--min_threshold=0.1",
    "--s_norm=2"
]

# 遍历number_of_adversaries的值，并执行每个命令
for aggregation_rule in aggregation_rules:
    for number_of_adversaries in numbers_of_adversaries:
        command = common_command + [f"--number_of_adversaries={number_of_adversaries}"] \
            + [f"--mal_boost={number_of_adversaries * 10}"] + [f"--aggregation_rule={aggregation_rule}"]
        print(f"Executing: {' '.join(command)}")
        subprocess.run(command, check=True)



# common_command = [
#     "python", "./FL_Backdoor_CV/roles/trainer.py",
#     "--dataset=fmnist",
#     "--params=utils/fmnist_params.yaml",
#     "--class_imbalance=0",
#     "--balance=0.99",
#     "--classes_per_client=2",
#     "--resume=1",
#     "--resumed_name=avg_MR_fmnist/avg_300.pth",
#     "--rounds=1000",
#     "--participant_population=500",
#     "--participant_sample_size=100",
#     "--number_of_adversaries=50",
#     "--mal_boost=500",
#     "--is_poison=1",
#     "--aggregation_rule=avg",
#     "--random_compromise=0",
#     "--retrain_rounds=60",
#     "--poison_prob=0.5",
#     "--device=cuda:0",
#     "--local_lr=0.1",
#     "--local_lr_decay=0.991",
#     "--decay_step=5",
#     "--local_lr_min=0.001",
#     "--global_lr=1",
#     "--global_lr_decay=1",
#     "--batch_size=64",
#     "--momentum=0.9",
#     "--decay=0.0005",
#     "--attack_mode=MR",
#     "--num_poisoned_samples=6",
#     "--gradmask_ratio=0.5",
#     "--multi_objective_num=4",
#     "--alternating_minimization=0",
#     "--record_step=100",
#     "--record_res_step=10",
#     "--weight=0.1",
#     "--min_threshold=0.1"
# ]

# subprocess.run(common_command, check=True)





# 定义mal_boost的值列表
# mal_boosts=[25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

# mal_boosts=[25, 150, 175, 200, 225, 250]


# # CIFAR10
# common_command = [
#     "python", "./FL_Backdoor_CV/roles/trainer.py",
#     "--dataset=cifar10",
#     "--params=utils/cifar10_params.yaml",
#     "--class_imbalance=0",
#     "--balance=0.99",
#     "--classes_per_client=2",
#     "--resume=1",
#     "--resumed_name=jzx_test_MR_cifar10/jzx_test_1000.pth",
#     "--rounds=1000",
#     "--participant_population=500",
#     "--participant_sample_size=100",
#     "--number_of_adversaries=20",
#     "--is_poison=1",
#     "--random_compromise=0",
#     "--retrain_rounds=60",
#     "--poison_prob=0.5",
#     "--aggregation_rule=jzx_test",
#     "--device=cuda:0",
#     "--local_lr=0.1",
#     "--local_lr_decay=0.991",
#     "--decay_step=5",
#     "--local_lr_min=0.001",
#     "--global_lr=1",
#     "--global_lr_decay=1",
#     "--batch_size=64",
#     "--momentum=0.9",
#     "--decay=0.0005",
#     "--attack_mode=MR",
#     "--num_poisoned_samples=6",
#     "--gradmask_ratio=0.5",
#     "--multi_objective_num=4",
#     "--alternating_minimization=0",
#     "--record_step=100",
#     "--record_res_step=10",
#     "--weight=0.03",
#     "--min_threshold=0.1"
# ]

# # 遍历mal_boost的值，并执行每个命令
# for mal_boost in mal_boosts:
#     command = common_command + [f"--mal_boost={mal_boost}"]
#     print(f"Executing: {' '.join(command)}")
#     subprocess.run(command, check=True)


mal_boosts=[125, 175, 225]

# FMNIST
common_command = [
    "python", "./FL_Backdoor_CV/roles/trainer.py",
    "--dataset=fmnist",
    "--params=utils/fmnist_params.yaml",
    "--class_imbalance=0",
    "--balance=0.99",
    "--classes_per_client=2",
    "--resume=1",
    "--resumed_name=avg_MR_fmnist/avg_300.pth",
    "--rounds=1000",
    "--participant_population=500",
    "--participant_sample_size=100",
    "--number_of_adversaries= 20",
    "--is_poison=1",
    "--random_compromise=0",
    "--retrain_rounds=60",
    "--poison_prob=0.5",
    "--aggregation_rule=jzx_test",
    "--device=cuda:0",
    "--local_lr=0.1",
    "--local_lr_decay=0.991",
    "--decay_step=5",
    "--local_lr_min=0.001",
    "--global_lr=1",
    "--global_lr_decay=1",
    "--batch_size=64",
    "--momentum=0.9",
    "--decay=0.0005",
    "--attack_mode=MR",
    "--num_poisoned_samples=6",
    "--gradmask_ratio=0.5",
    "--multi_objective_num=4",
    "--alternating_minimization=0",
    "--record_step=100",
    "--record_res_step=10",
    "--weight=0.1",
    "--min_threshold=0.1"
]

# 遍历mal_boost的值，并执行每个命令
for mal_boost in mal_boosts:
    command = common_command + [f"--mal_boost={mal_boost}"]
    print(f"Executing: {' '.join(command)}")
    subprocess.run(command, check=True)



mal_boosts=[25, 50, 75, 100, 125, 150, 175, 200, 225, 250]


# CIFAR10
common_command = [
    "python", "./FL_Backdoor_CV/roles/trainer.py",
    "--dataset=cifar10",
    "--params=utils/cifar10_params.yaml",
    "--class_imbalance=0",
    "--balance=0.99",
    "--classes_per_client=2",
    "--resume=1",
    "--resumed_name=jzx_test_MR_cifar10/jzx_test_1000.pth",
    "--rounds=1000",
    "--participant_population=500",
    "--participant_sample_size=100",
    "--number_of_adversaries= 20",
    "--is_poison=1",
    "--random_compromise=0",
    "--retrain_rounds=60",
    "--poison_prob=0.5",
    "--aggregation_rule=avg",
    "--device=cuda:0",
    "--local_lr=0.1",
    "--local_lr_decay=0.991",
    "--decay_step=5",
    "--local_lr_min=0.001",
    "--global_lr=1",
    "--global_lr_decay=1",
    "--batch_size=64",
    "--momentum=0.9",
    "--decay=0.0005",
    "--attack_mode=MR",
    "--num_poisoned_samples=6",
    "--gradmask_ratio=0.5",
    "--multi_objective_num=4",
    "--alternating_minimization=0",
    "--record_step=100",
    "--record_res_step=10",
    "--weight=0.1",
    "--min_threshold=0.1",
    "--s_norm=2"
]

# 遍历mal_boost的值，并执行每个命令
for mal_boost in mal_boosts:
    command = common_command + [f"--mal_boost={mal_boost}"]
    print(f"Executing: {' '.join(command)}")
    subprocess.run(command, check=True)



# FMNIST
common_command = [
    "python", "./FL_Backdoor_CV/roles/trainer.py",
    "--dataset=fmnist",
    "--params=utils/fmnist_params.yaml",
    "--class_imbalance=0",
    "--balance=0.99",
    "--classes_per_client=2",
    "--resume=1",
    "--resumed_name=avg_MR_fmnist/avg_300.pth",
    "--rounds=1000",
    "--participant_population=500",
    "--participant_sample_size=100",
    "--number_of_adversaries= 20",
    "--is_poison=1",
    "--random_compromise=0",
    "--retrain_rounds=60",
    "--poison_prob=0.5",
    "--aggregation_rule=avg",
    "--device=cuda:0",
    "--local_lr=0.1",
    "--local_lr_decay=0.991",
    "--decay_step=5",
    "--local_lr_min=0.001",
    "--global_lr=1",
    "--global_lr_decay=1",
    "--batch_size=64",
    "--momentum=0.9",
    "--decay=0.0005",
    "--attack_mode=MR",
    "--num_poisoned_samples=6",
    "--gradmask_ratio=0.5",
    "--multi_objective_num=4",
    "--alternating_minimization=0",
    "--record_step=100",
    "--record_res_step=10",
    "--weight=0.1",
    "--min_threshold=0.1"
]

# 遍历mal_boost的值，并执行每个命令
for mal_boost in mal_boosts:
    command = common_command + [f"--mal_boost={mal_boost}"]
    print(f"Executing: {' '.join(command)}")
    subprocess.run(command, check=True)




# CIFAR10
common_command = [
    "python", "./FL_Backdoor_CV/roles/trainer.py",
    "--dataset=cifar10",
    "--params=utils/cifar10_params.yaml",
    "--class_imbalance=0",
    "--balance=0.99",
    "--classes_per_client=2",
    "--resume=1",
    "--resumed_name=jzx_test_MR_cifar10/jzx_test_1000.pth",
    "--rounds=1000",
    "--participant_population=500",
    "--participant_sample_size=100",
    "--number_of_adversaries= 20",
    "--is_poison=1",
    "--random_compromise=0",
    "--retrain_rounds=60",
    "--poison_prob=0.5",
    "--aggregation_rule=flame",
    "--device=cuda:0",
    "--local_lr=0.1",
    "--local_lr_decay=0.991",
    "--decay_step=5",
    "--local_lr_min=0.001",
    "--global_lr=1",
    "--global_lr_decay=1",
    "--batch_size=64",
    "--momentum=0.9",
    "--decay=0.0005",
    "--attack_mode=MR",
    "--num_poisoned_samples=6",
    "--gradmask_ratio=0.5",
    "--multi_objective_num=4",
    "--alternating_minimization=0",
    "--record_step=100",
    "--record_res_step=10",
    "--weight=0.1",
    "--min_threshold=0.1"
]

# 遍历mal_boost的值，并执行每个命令
for mal_boost in mal_boosts:
    command = common_command + [f"--mal_boost={mal_boost}"]
    print(f"Executing: {' '.join(command)}")
    subprocess.run(command, check=True)



# # FMNIST
# common_command = [
#     "python", "./FL_Backdoor_CV/roles/trainer.py",
#     "--dataset=fmnist",
#     "--params=utils/fmnist_params.yaml",
#     "--class_imbalance=0",
#     "--balance=0.99",
#     "--classes_per_client=2",
#     "--resume=1",
#     "--resumed_name=avg_MR_fmnist/avg_300.pth",
#     "--rounds=1000",
#     "--participant_population=500",
#     "--participant_sample_size=100",
#     "--number_of_adversaries= 20",
#     "--is_poison=1",
#     "--random_compromise=0",
#     "--retrain_rounds=60",
#     "--poison_prob=0.5",
#     "--aggregation_rule=flame",
#     "--device=cuda:0",
#     "--local_lr=0.1",
#     "--local_lr_decay=0.991",
#     "--decay_step=5",
#     "--local_lr_min=0.001",
#     "--global_lr=1",
#     "--global_lr_decay=1",
#     "--batch_size=64",
#     "--momentum=0.9",
#     "--decay=0.0005",
#     "--attack_mode=MR",
#     "--num_poisoned_samples=6",
#     "--gradmask_ratio=0.5",
#     "--multi_objective_num=4",
#     "--alternating_minimization=0",
#     "--record_step=100",
#     "--record_res_step=10",
#     "--weight=0.1",
#     "--min_threshold=0.1"
# ]

# # 遍历mal_boost的值，并执行每个命令
# for mal_boost in mal_boosts:
#     command = common_command + [f"--mal_boost={mal_boost}"]
#     print(f"Executing: {' '.join(command)}")
#     subprocess.run(command, check=True)



# s_norms = [2, 3, 4, 5]

# # FMNIST
# common_command = [
#     "python", "./FL_Backdoor_CV/roles/trainer.py",
#     "--dataset=fmnist",
#     "--params=utils/fmnist_params.yaml",
#     "--class_imbalance=0",
#     "--balance=0.99",
#     "--classes_per_client=2",
#     "--resume=1",
#     "--resumed_name=avg_MR_fmnist/avg_300.pth",
#     "--rounds=1000",
#     "--participant_population=500",
#     "--participant_sample_size=100",
#     "--number_of_adversaries= 50",
#     "--is_poison=1",
#     "--mal_boost=500",
#     "--random_compromise=0",
#     "--retrain_rounds=60",
#     "--poison_prob=0.5",
#     "--aggregation_rule=avg",
#     "--device=cuda:0",
#     "--local_lr=0.1",
#     "--local_lr_decay=0.991",
#     "--decay_step=5",
#     "--local_lr_min=0.001",
#     "--global_lr=1",
#     "--global_lr_decay=1",
#     "--batch_size=64",
#     "--momentum=0.9",
#     "--decay=0.0005",
#     "--attack_mode=MR",
#     "--num_poisoned_samples=6",
#     "--gradmask_ratio=0.5",
#     "--multi_objective_num=4",
#     "--alternating_minimization=0",
#     "--record_step=100",
#     "--record_res_step=10",
#     "--weight=0.1",
#     "--min_threshold=0.1"
# ]

# # 遍历mal_boost的值，并执行每个命令
# for s_norm in s_norms:
#     command = common_command + [f"--s_norm={s_norm}"]
#     print(f"Executing: {' '.join(command)}")
#     subprocess.run(command, check=True)