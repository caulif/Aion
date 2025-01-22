import pickle
import os
import random
from decimal import Decimal, getcontext

getcontext().prec = 1024  # 设置十进制运算精度为 1024 位

class SHPRG:  # 将类名更改为 SHPRF
    """
    安全伪随机函数 (Secure Pseudo-Random Function) 类
    """
    def __init__(self, n, m, p, q, filename):
        """
        初始化 SHPRF 对象

        参数：
            n (int): 矩阵的行数
            m (int): 矩阵的列数
            p (int): 模运算的第一个参数
            q (int): 模运算的第二个参数
            filename (str): 存储矩阵的文件名
        """
        assert p < q, "p < q"  # 确保 p 小于 q
        assert n < m, "n < m"  # 确保 n 小于 m

        self.n = n
        self.m = m
        self.p = p
        self.q = q
        self.filename = filename

        # 如果文件存在，则加载矩阵
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                self.A = pickle.load(file)
        # 否则，生成一个新的随机矩阵并保存到文件
        else:
            self.A = [[random.randint(0, q - 1) for _ in range(m)] for _ in range(n)]
            with open(filename, 'wb') as file:
                pickle.dump(self.A, file)

        # # 将a取模
        # for i in range(n):
        #     for j in range(m):
        #         self.A[i][j] = self.A[i][j] % q
        # print(self.A)

    def G(self, s):
        """
        计算矩阵乘积 A^T * s，并将结果映射到 [0, p) 范围内

        参数：
            s (int): 种子值

        返回值：
            list: 映射后的结果向量
        """
        # 将 s 转换为列向量形式
        s = [[s]]
        # 计算 A^T * s
        product = []
        for j in range(self.m):
            sum_result = 0
            for i in range(self.n):
                sum_result += self.A[i][j] * s[i][0]
            product.append(sum_result % self.q)  # 模运算

        # 将结果转换为十进制数
        product = [Decimal(product[j]) for j in range(self.m)]
        p = Decimal(self.p)
        q = Decimal(self.q)

        # 将结果映射到 [0, p) 范围内
        result = [int((x * p / q + Decimal('0.5'))) for x in product]
        # print("G(s):", result)
        return result

    def hprf(self, k, x, length):
        result = []
        counter = 0
        while len(result) < length:
            #  关键修改：将计数器与 x 结合后再与 k 相乘
            s = (k * (x + counter)) % self.q
            result.extend(self.G(s))
            counter += 1
        return result[:length]


    def list_hprf(self, k, x, length):
        result = []
        counter = 0
        offset = k[0][0]  # 获取偏移量
        initial_value = k[0][1]  # 获取初始值

        while len(result) < length:
            # 使用初始值和计数器进行计算
            s = (initial_value * (x + counter)) % self.q
            # print("s:", s)
            generated_values = self.G(s)

            # 为生成的每个值添加偏移量并转换为元组
            for val in generated_values:
                result.append((offset, val))

            counter += 1
        return result[:length]


    def hprg(self, seed, length):  # 保留此函数用于根据种子生成多个值，但它不是 PRF 功能的核心。
        """
         生成指定长度的伪随机数序列

         参数：
             seed (int): 种子值
             length (int): 序列长度

         返回值：
             list: 伪随机数序列
         """
        s = seed
        extended_vector = self.G(s)  # 计算初始向量
        if length > self.m:
            # 如果序列长度大于矩阵列数，则重复扩展初始向量
            repeated_vector = (extended_vector * (length // self.m + 1))[:length]
            return repeated_vector
        else:
            # 否则，返回初始向量的前 length 个元素
            return extended_vector[:length]


    def list_hprg(self, seed, length):
        """
        生成指定长度的伪随机数序列

        参数：
            seed (int): 种子值
            length (int): 序列长度

        返回值：
            list: 伪随机数序列
        """
        # s = seed
        result = []
        offset = seed[0][0]  # 获取偏移量
        initial_value = seed[0][1]  # 获取初始值
        extended_vector = self.G(initial_value)  # 计算初始向量
        if length > self.m:
            # 如果序列长度大于矩阵列数，则重复扩展初始向量
            repeated_vector = (extended_vector * (length // self.m + 1))[:length]
            for i in range(len(repeated_vector)):
                result.append((offset, repeated_vector[i]))
            return result
        else:
            # 否则，返回初始向量的前 length 个元素
            for i in range(len(extended_vector)):
                result.append((offset, extended_vector[i]))
            return result[:length]


def load_initialization_values(filename):
    """
    从文件中加载初始化值

    参数：
        filename (str): 文件名

    返回值：
        tuple: 初始化值 (n, m, p, q)
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)