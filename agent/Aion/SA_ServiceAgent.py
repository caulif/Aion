import time
import logging
import random
import json
import math
import multiprocessing
import dill
import re
import pickle

import pandas as pd
import numpy as np
import sympy

from Crypto.Hash import SHA256
from Crypto.Signature import DSS


from agent.Agent import Agent  # 导入 Agent 基类
from agent.HPRF.hprf import load_initialization_values, SHPRG  # 导入 HPRF 相关模块
from agent.Aion.tool import Tool  # 导入 Flamingo 工具模块
from message.Message import Message  # 导入消息类，用于代理之间通信
from util import param, util  # 导入参数设置和实用函数模块


# PPFL_ServiceAgent 类继承自基础 Agent 类。
class SA_ServiceAgent(Agent):
    """
    表示参与安全聚合协议的服务器代理。
    """

    def __str__(self):
        return "[server]"

    def __init__(self, id, name, type,
                 random_state=None,
                 msg_fwd_delay=1_000_000,  # 转发点对点客户端消息的延迟（以纳秒为单位）
                 round_time=pd.Timedelta("10s"),
                 iterations=4,
                 key_length=32,
                 num_clients=10,
                 parallel_mode=1,
                 debug_mode=0,
                 Dimension=10_000,
                 commit_size=8,
                 msg_name=None,
                 users=None):
        """
        初始化服务器代理。

        Args:
            id (int): 代理的唯一 ID。
            name (str): 代理的名称。
            type (str): 代理的类型。
            random_state (random.Random, optional): 随机数生成器。默认为 None。
            msg_fwd_delay (int, optional): 转发点对点客户端消息的延迟（以纳秒为单位）。默认为 1000000。
            round_time (pandas.Timedelta, optional): 每轮的默认等待时间。默认为 "10s"。
            iterations (int, optional): 协议的迭代次数。默认为 4。
            key_length (int, optional): 加密密钥的长度（以字节为单位）。默认为 32。
            num_clients (int, optional): 参与协议的客户端数量。默认为 10。
            parallel_mode (int, optional): 是否启用并行模式。默认为 1。
            debug_mode (int, optional): 是否启用调试模式。默认为 0。
            Dimension (int, optional): 向量维度。默认为 10000
            commit_size (int, optional): 委员会大小。默认为 8
            msg_name (str, optional): 消息名称。默认为 None
            users (set, optional): 用户 ID 集合。默认为空集合。
        """
        super().__init__(id, name, type, random_state)  # 调用父类初始化方法

        # 设置日志记录器
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # 设置日志级别
        if debug_mode:
            logging.basicConfig(level=logging.DEBUG)  # 启用调试模式

        # 系统参数
        self.msg_fwd_delay = msg_fwd_delay  # 转发点对点客户端消息的时间
        self.round_time = round_time  # 默认每轮的等待时间
        self.no_of_iterations = iterations  # 迭代次数
        self.parallel_mode = parallel_mode  # 并行模式

        # 输入参数
        self.num_clients = num_clients  # 每轮训练的用户数量
        self.users = users if users is not None else set()  # 用户 ID 列表
        self.vector_len = Dimension  # 向量长度
        self.vector_dtype = param.vector_type  # 向量数据类型

        # 安全参数
        self.commit_size = commit_size
        self.msg_name = msg_name
        self.committee_threshold = 0  # 委员会阈值
        self.prime = param.prime  # 使用的素数

        # 数据存储
        self.times = []  # 用于存储各轮迭代用时
        self.client_id_list = None  # 客户端 ID 列表
        self.seed_sum_hprf = None  # 掩码种子和
        self.selected_indices = None  # 选择的客户端索引
        self.committee_shares_sum = None  # 委员会解密份额之和
        self.seed_sum = None  # 掩码种子和
        self.recv_user_masked_vectors = {}  # 存储接收到的客户端掩码向量
        self.recv_committee_shares_sum = {}  # 接收到的委员会解密份额之和
        self.user_masked_vectors = {}  # 用户向量池
        self.user_committee = {}  # 用户委员会池
        self.committee_sigs = {}  # 委员会签名池
        self.recv_committee_sigs = {}  # 接收到的委员会签名池
        self.receive_mask = {}  # 接收到的掩码池

        # 初始化向量
        self.vec_sum_partial = np.zeros(self.vector_len, dtype=self.vector_dtype)  # 部分向量和
        self.final_sum = np.zeros(self.vector_len, dtype=self.vector_dtype)  # 最终向量和

        # 跟踪当前协议的迭代和轮次
        self.current_iteration = 1  # 当前迭代次数
        self.current_round = 0  # 当前轮次

        # 用于防投毒模块的参数
        self.l2_old = []  # 初始化l2_old列表
        self.linf_old = 0.1  # 初始化linf_old
        self.linf_shprg_old = 0.05  # 初始化linf_shprg_old
        self.b_old = 0.2  # 初始化b_old

        # 累积时间
        self.elapsed_time = {
            'REPORT': pd.Timedelta(0),
            'CROSSCHECK': pd.Timedelta(0),
            'RECONSTRUCTION': pd.Timedelta(0),
        }

        # 消息处理映射
        self.aggProcessingMap = {
            0: self.initialize,  # 初始化
            1: self.report,  # 报告
            2: self.forward_signatures,  # 转发签名
            3: self.reconstruction,  # 重建
        }

        # 轮次名称映射
        self.namedict = {
            0: "initialize",
            1: "report",
            2: "forward_signatures",
            3: "reconstruction",
        }

        # 记录各部分时间
        self.timings = {
            "seed sharing": [],
            "Legal clients confirmation": [],
            "Masked model generation": [],
            "Online clients confirmation": [],
            "Aggregate share reconstruction": [],
            "Model aggregation": [],
        }

    # 模拟生命周期消息
    def kernelStarting(self, startTime):
        """
        在内核启动时初始化服务器状态。

        Args:
            startTime (pandas.Timestamp): 内核启动时间。
        """
        self.starttime = time.time()
        self.kernel.custom_state['srv_report'] = pd.Timedelta(0)  # 报告时间
        self.kernel.custom_state['srv_crosscheck'] = pd.Timedelta(0)  # 交叉检查时间
        self.kernel.custom_state['srv_reconstruction'] = pd.Timedelta(0)  # 重建时间
        self.setComputationDelay(0)  # 设置计算延迟为 0
        super().kernelStarting(startTime)  # 调用父类方法

    def kernelStopping(self):
        """
        在内核停止时进行清理工作，包括计算平均时间。
        """
        # 计算平均时间并记录到内核状态
        self.kernel.custom_state['srv_report'] += (self.elapsed_time['REPORT'] / self.no_of_iterations)
        self.kernel.custom_state['srv_crosscheck'] += (self.elapsed_time['CROSSCHECK'] / self.no_of_iterations)
        self.kernel.custom_state['srv_reconstruction'] += (self.elapsed_time['RECONSTRUCTION'] / self.no_of_iterations)

        # 打印总时间和各部分时间
        self.stoptime = time.time()


        print("各部分用时统计:")
        for part, times in self.timings.items():
            if times:
                avg_time = sum(times) / len(times)
                print(f"{part}: 平均用时 {avg_time:.6f} 秒, 总用时 {sum(times):.6f} 秒")


        super().kernelStopping()  # 调用父类方法

    def wakeup(self, currentTime):
        """
        在每轮结束时被调用，根据当前轮次执行相应的处理。

        Args:
            currentTime (pandas.Timestamp): 当前模拟时间。
        """
        super().wakeup(currentTime)  # 调用父类方法
        self.agent_print(
            f"wakeup in iteration {self.current_iteration} at function {self.namedict[self.current_round]}; current time is {currentTime}")
        self.aggProcessingMap[self.current_round](currentTime)  # 根据当前轮次调用相应的处理函数

    def receiveMessage(self, currentTime, msg):
        """
        接收消息并存储。

        Args:
            currentTime (pandas.Timestamp): 当前时间。
            msg (Message): 收到的消息。
        """
        super().receiveMessage(currentTime, msg)  # 调用父类方法
        sender_id = msg.body['sender']  # 获取发送者 ID

        if msg.body['msg'] == "VECTOR" and msg.body['iteration'] == self.current_iteration:  # 接收掩码向量
            self.recv_user_masked_vectors[sender_id] = msg.body['masked_vector']

        elif msg.body['msg'] == "SIGN" and msg.body['iteration'] == self.current_iteration:  # 接收签名
            self.recv_committee_sigs[sender_id] = msg.body['signed_labels']

        elif msg.body['msg'] == "hprf_SUM_SHARES" and msg.body['iteration'] == self.current_iteration:  # 接收秘密份额
            self.recv_committee_shares_sum[sender_id] = msg.body['sum_shares']

        elif msg.body['msg'] == "BFT_SIGN" and msg.body['iteration'] == self.current_iteration:
            self.recv_committee_sigs[sender_id] = msg.body['signed_labels']

    def initialize(self, currentTime):
        """
        初始化协议，包括选择委员会成员和发送初始模型。

        Args:
            currentTime (pandas.Timestamp): 当前模拟时间。
        """
        start_time = time.time()
        dt_protocol_start = pd.Timestamp('now')  # 记录协议开始时间
        self.user_committee = param.choose_committee(param.root_seed, self.commit_size, self.num_clients)  # 选择委员会成员
        self.committee_threshold = len(self.user_committee) // 4  # 计算委员会阈值，需要接收至少 1/4 的份额

        initial_model_weights = np.ones(self.vector_len, dtype=self.vector_dtype) * 1000
        message = Message({"msg": "INITIAL_MODEL", "iteration": 0, "model_weights": initial_model_weights})  # 创建初始化模型消息

        self.timings["seed sharing"].append(time.time() - start_time)  # 记录种子分享时间
        self.client_id_list = [i for i in range(self.num_clients)]
        start_time = time.time()
        self.recv_committee_sigs = self.BFT_broadcast(message, self.client_id_list)  # 广播消息，并等待确认
        self.timings["Legal clients confirmation"].append(time.time() - start_time)  # 记录合法客户端确认时间

        self.current_round = 1  # 进入报告阶段
        server_comp_delay = pd.Timestamp('now') - dt_protocol_start  # 计算服务器计算延迟
        self.setWakeup(currentTime + server_comp_delay + pd.Timedelta('2s'))  # 设置下次唤醒时间

    def BFT_broadcast(self, message, client_ids):
        """
        将消息广播给指定的客户端并等待响应。

        Args:
            message (Message): 要广播的消息。
            client_ids (list): 目标客户端 ID 列表。
        Returns:
            dict: 返回接收到的签名消息
        """
        for client_id in client_ids:
            self.sendMessage(client_id, message, tag="comm_dec_server", msg_name=self.msg_name)  # 发送消息

        start_time = time.time()
        Delta = pd.Timedelta('0.005s')
        while time.time() - start_time < 2 * Delta.total_seconds():
            if len(self.recv_committee_sigs) == len(client_ids):
                break
            time.sleep(0.0001)

        return self.recv_committee_sigs  # 返回接收到的签名

    def report(self, currentTime):
        """
        处理客户端的掩码向量，并发送解密请求。

        Args:
            currentTime (pandas.Timestamp): 当前模拟时间。
        """
        start_time = time.time()
        self.report_time = time.time()
        dt_protocol_start = pd.Timestamp('now')  # 记录协议开始时间

        self.report_read_from_pool()  # 从接收池中读取数据
        self.report_process()  # 处理数据，计算部分和
        self.report_clear_pool()  # 清空消息池

        # BFT 确认在线客户端信息
        online_clients_list = [1 if i in self.recv_user_masked_vectors else 0 for i in range(self.num_clients)]
        online_clients_list_bytes = bytes(online_clients_list)
        message_online_clients = Message({"msg": "ONLINE_CLIENTS", "iteration": self.current_iteration,
                                          "online_clients": online_clients_list_bytes})
        self.recv_committee_sigs = self.BFT_broadcast(message_online_clients, self.user_committee)  # BFT 广播

        self.report_send_message()  # 发送解密请求

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start  # 计算服务器计算延迟
        self.agent_print("报告步骤的运行时间:", server_comp_delay)  # 打印调试信息

        self.recordTime(dt_protocol_start, "REPORT")  # 记录报告时间
        self.report_time = time.time() - self.report_time  # 记录报告时间

        self.current_round = 2  # 进入转发签名阶段
        self.setWakeup(currentTime + server_comp_delay)  # 设置下次唤醒时间
        self.timings["Masked model generation"].append(time.time() - start_time)

    def report_read_from_pool(self):
        """从接收池中读取数据。"""
        self.user_masked_vectors = self.recv_user_masked_vectors
        self.recv_user_masked_vectors = {}

    def report_clear_pool(self):
        """清空消息池。"""
        self.recv_committee_shares_sum = {}
        self.recv_committee_sigs = {}

    def report_process(self):
        """
        处理掩码向量，计算部分和。
        """
        self.agent_print("收集到的向量数量:", len(self.user_masked_vectors))  # 打印调试信息

        self.client_id_list = list(self.user_masked_vectors.keys())  # 获取所有客户端 ID

        self.selected_indices, self.b_old = self.MMF(self.user_masked_vectors, self.l2_old, self.linf_old,
                                                     self.linf_shprg_old, self.b_old,
                                                     self.current_iteration)  # 使用 MMF 选择客户端

        self.vec_sum_partial = np.zeros(self.vector_len, np.int64)  # 初始化部分和
        for id in self.selected_indices:  # 计算选定客户端的掩码向量部分和
            if len(self.user_masked_vectors[id]) != self.vector_len:
                raise RuntimeError("客户端发送了不正确长度的向量。")
            self.vec_sum_partial += self.user_masked_vectors[id]
            self.vec_sum_partial %= self.prime

    def report_send_message(self):
        """向委员会成员发送解密请求。"""
        for id in self.user_committee:  # 循环发送解密请求给每个委员会成员
            self.sendMessage(id,
                             Message({"msg": "SIGN",
                                      "iteration": self.current_iteration,
                                      "client_id_list": self.client_id_list,
                                      }),
                             tag="comm_dec_server",
                             msg_name=self.msg_name)

    def forward_signatures(self, currentTime):
        """
        转发签名并请求秘密份额。

        Args:
            currentTime (pandas.Timestamp): 当前模拟时间。
        """
        dt_protocol_start = pd.Timestamp('now')  # 记录协议开始时间
        self.check_time = time.time()

        for id in self.user_committee:  # 向委员会成员发送请求秘密份额的消息
            self.sendMessage(id,
                             Message({"msg": "request shares sum",
                                      "iteration": self.current_iteration,
                                      "request id list": self.selected_indices,
                                      }),
                             tag="comm_sign_server",
                             msg_name=self.msg_name)

        self.current_round = 3  # 进入重构阶段
        server_comp_delay = pd.Timestamp('now') - dt_protocol_start  # 计算服务器计算延迟
        self.agent_print("交叉检查步骤的运行时间:", server_comp_delay)  # 打印调试信息
        self.setWakeup(currentTime + server_comp_delay + param.wt_flamingo_reconstruction)  # 设置下次唤醒时间

        self.recordTime(dt_protocol_start, "CROSSCHECK")  # 记录交叉检查时间
        self.check_time = time.time() - self.check_time

    def reconstruction(self, currentTime):
        """
        执行向量重构，并发送最终结果给客户端。

        Args:
            currentTime (pandas.Timestamp): 当前模拟时间。
        """
        dt_protocol_start = pd.Timestamp('now')  # 记录协议开始时间
        self.reco_time = time.time()

        self.reconstruction_read_from_pool()  # 从接收池中读取数据
        self.reconstruction_process()  # 处理数据，重建向量和
        self.reconstruction_clear_pool()  # 清空消息池
        self.reconstruction_send_message()  # 发送最终结果

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start  # 计算服务器计算延迟
        self.agent_print("Reconstruction time:", server_comp_delay)  # 打印调试信息
        self.recordTime(dt_protocol_start, "RECONSTRUCTION")  # 记录重构时间
        self.reco_time = time.time() - self.reco_time  # 记录重构时间

        print()
        print("######## Iteration completed ########")  # 打印迭代完成信息
        print(f"[Server] Completed iteration {self.current_iteration} at {currentTime + server_comp_delay}")
        print()

        self.current_round = 1  # 进入下一轮报告阶段
        self.current_iteration += 1  # 进入下一轮迭代
        if self.current_iteration > self.no_of_iterations:
            return  # 结束模拟

        self.setWakeup(currentTime + server_comp_delay + param.wt_flamingo_report)  # 设置下次唤醒时间

    def reconstruction_read_from_pool(self):
        """从接收池中读取解密份额。"""
        while len(self.recv_committee_shares_sum) < self.committee_threshold:
            time.sleep(0.01)  # 等待，直到收到足够的份额

        self.committee_shares_sum = self.recv_committee_shares_sum  # 将接收到的份额移动到处理池
        self.recv_committee_shares_sum = {}

    def reconstruction_clear_pool(self):
        """清空所有消息池。"""
        self.user_masked_vectors = {}
        self.committee_shares_sum = {}

        self.recv_user_masked_vectors = {}
        self.recv_committee_shares_sum = {}
        self.recv_user_masked_vectors = {}

    def reconstruction_process(self):
        """处理秘密份额，重建掩码和最终向量和。"""
        self.agent_print("来自解密器的收集份额数量:", len(self.committee_shares_sum))  # 打印调试信息
        if len(self.committee_shares_sum) < self.committee_threshold:
            raise RuntimeError("未收到足够的解密份额。")  # 抛出异常

        start_time = time.time()
        committee_shares_sum_list = [shares_list for _, shares_list in self.committee_shares_sum.items()]
        self.seed_sum_hprf = SA_ServiceAgent.reconstruct_secret_vector(committee_shares_sum_list, self.prime)  # 重建掩码种子和
        self.timings["Aggregate share reconstruction"].append(time.time() - start_time)

        start_time = time.time()
        self.final_sum = self.vec_sum_partial - self.seed_sum_hprf  # 重建最终向量和
        self.final_sum //= len(self.selected_indices)  # 除以参与客户端数量
        self.final_sum = np.array(self.final_sum, dtype=np.uint32)
        self.final_sum %= self.prime


        self.l2_old = [np.linalg.norm(self.final_sum)] + self.l2_old[:1]  # 更新l2_old，只保留最新的两个值
        self.linf_old = np.max(np.abs(self.final_sum))  # 更新linf_old
        self.linf_shprg_old = np.max(np.abs(self.seed_sum_hprf))  # 更新linf_shprg_old
        message_final_sum = Message(
            {"msg": "FINAL_SUM", "iteration": self.current_iteration, "final_sum": self.final_sum})

        self.timings["Model aggregation"].append(time.time() - start_time)
        start_time = time.time()
        self.BFT_broadcast(message_final_sum, self.user_committee)  # BFT 广播最终结果
        self.timings["Online clients confirmation"].append(time.time() - start_time)

    @staticmethod
    def reconstruct_secret(shares: list, prime: int) -> int:
        """
        从秘密分享中恢复秘密值。

        Args:
            shares (list): 秘密分享列表。
            prime (int): 使用的素数。

        Returns:
            int: 恢复的秘密值。
        """
        secret = 0
        shares_list = [shares_list_for_client[0] for _, shares_list_for_client in shares.items()]
        for i, (x_i, y_i) in enumerate(shares_list):
            numerator, denominator = 1, 1
            for j, (x_j, _) in enumerate(shares_list):
                if i == j:
                    continue
                numerator *= -x_j
                denominator *= (x_i - x_j)
            lagrange_coefficient = (numerator * sympy.mod_inverse(denominator, prime)) % prime
            secret = (secret + y_i * lagrange_coefficient) % prime
        return secret

    @staticmethod
    def reconstruct_secret_vector(shares: list, prime: int) -> list:
        """
        从秘密分享中恢复秘密向量，所有元素共享相同的拉格朗日系数.

        Args:
            shares (list): 每个向量元素的秘密分享列表。
            prime (int): 使用的素数。

        Returns:
            list: 恢复的秘密向量。
        """
        n = len(shares)  # 向量的维度
        k = len(shares[0])  # 用于重建的份额数量（假设所有元素的份额数量相同）

        # 预先计算拉格朗日系数，因为它们对所有元素都相同
        lagrange_coefficients = []
        for i in range(n):
            numerator, denominator = 1, 1
            x_i, _ = shares[i][0]  # 取第一个元素的 x 值，因为 x 值对所有元素相同
            for j in range(n):
                if i == j:
                    continue
                x_j, _ = shares[j][0]
                numerator = (numerator * (-1 * x_j)) % prime
                denominator = (denominator * (x_i - x_j)) % prime
            lagrange_coefficients.append((numerator * sympy.mod_inverse(denominator, prime)) % prime)

        secret_vector = []
        for i in range(k):  # 遍历每个元素的份额
            secret_element = 0
            for j in range(n):
                _, y_i = shares[j][i]
                secret_element = (secret_element + y_i * lagrange_coefficients[j]) % prime  # 使用预先计算的拉格朗日系数
            secret_vector.append(secret_element % prime)
        return secret_vector

    @staticmethod
    def vss_reconstruct(shares: list, prime: int) -> int:
        """
        本地恢复秘密函数

        Args:
            shares (list): 秘密分享的列表
            prime (int): 使用的素数

        Returns:
            int: 恢复的秘密
        """
        return SA_ServiceAgent.reconstruct_secret(shares, prime)

    def reconstruction_send_message(self):
        """发送最终结果给客户端。"""
        for id in self.users:
            self.sendMessage(id,
                             Message({"msg": "REQ", "sender": 0, "output": 1}),
                             tag="comm_output_server",
                             msg_name=self.msg_name)

    def MMF(self, masked_updates, l2_old, linf_old, linf_shprg_old, b_old, current_round):
        """
        选择良性客户端的函数。

        Args:
            masked_updates (dict):  客户端masked更新的字典,键为客户端索引(int)，值为numpy一维数组。
            ... (其他参数不变)

        Returns:
            list: 良性客户端的索引列表.
            float: 当前轮的阈值b
        """
        WEIGHT = 1.0  # 权重
        MIN_THRESHOLD = 0.3  # 最小阈值
        RESUME = False  # 是否恢复训练
        RESUMED_NAME = None  # 恢复训练的文件名

        cnt = len(masked_updates)  # 获取客户端数量


        # 计算L2范数并排序，保留原始索引
        l2_norm = {k: np.linalg.norm(v) for k, v in masked_updates.items()}
        sorted_l2_norm = dict(sorted(l2_norm.items(), key=lambda item: item[1]))

        # 动态调整阈值b (逻辑不变)
        if current_round <= 3 or (
                RESUME and current_round <= int(
                    re.findall(r'\d+\d*', RESUMED_NAME.split('/')[1])[0]) + 3 if RESUMED_NAME else 0):
            b = list(sorted_l2_norm.values())[int(MIN_THRESHOLD * cnt)]
        else:
            b = (l2_old[1] + linf_shprg_old) / (l2_old[0] + linf_shprg_old) * b_old

        # 选择良性客户端，使用排序后的字典
        selected_indices = []
        count = 0
        for k, v in sorted_l2_norm.items():
            if v <= b:
                selected_indices.append(k)
                count += 1
            if count >= int(0.8 * cnt):  # 这里也需要限制最大数量
                break

        benign_index = max(int(MIN_THRESHOLD * cnt),
                           min(int(0.8 * cnt), len(selected_indices)))  # 保证最少选择MIN_THRESHOLD * cnt个客户端
        if len(selected_indices) > benign_index:
            selected_indices = selected_indices[:benign_index]
        else:
            selected_indices = list(sorted_l2_norm.keys())[:benign_index]
        return selected_indices, b

    # ======================== UTIL ========================
    def recordTime(self, startTime, categoryName):
        """
        记录时间。

        Args:
            startTime (pandas.Timestamp): 开始时间。
            categoryName (str): 类别名称。
        """
        dt_protocol_end = pd.Timestamp('now')  # 获取当前时间
        self.elapsed_time[categoryName] += dt_protocol_end - startTime  # 计算经过时间并累加到对应类别

    def agent_print(*args, **kwargs):
        """
        自定义打印函数，在打印之前添加 [Server] 标头。
        """
        print(f"[Server] ", *args, **kwargs)  # 打印信息