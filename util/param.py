import sympy
from Cryptodome.Cipher import AES, ChaCha20
from Cryptodome.Random import get_random_bytes
import numpy as np
import pandas as pd
import math

# System parameters
vector_len = 10000
# modified by czh, 2024.05.11=2
# TODO 自动读取、更新模型参数
# vector_len = 44426
vector_type = 'uint32'
# modified by cx 2024.05.21 60->10
committee_size = 8
fraction = 1/3
fixed_key = b"abcd"

# prime = sympy.randprime(2 ** (32 - 1), 2 ** 32)  # 生成一个大素数
prime = 527921
# Waiting time
# Set according to a target dropout rate (e.g., 1%) 
# and message lantecy (see model/LatencyModel.py)
wt_flamingo_report = pd.Timedelta('10s')
wt_flamingo_crosscheck = pd.Timedelta('3s')
wt_flamingo_reconstruction = pd.Timedelta('3s')

wt_google_adkey = pd.Timedelta('10s')
wt_google_graph = pd.Timedelta('10s')
wt_google_share = pd.Timedelta('30s')    # ensure all user_choice received messages
wt_google_collection = pd.Timedelta('10s')
wt_google_crosscheck = pd.Timedelta('3s')
wt_google_recontruction = pd.Timedelta('2s') 

# WARNING: 
# this should be a random seed from beacon service;
# we use a fixed one for simplicity
root_seed = get_random_bytes(32) 
nonce = b'\x00\x00\x00\x00\x00\x00\x00\x00'

# choose committee members
def choose_committee(root_seed, committee_size, num_clients):
    """
    从所有客户端中选择一个委员会，用于执行安全聚合过程中的解密步骤。

    Args:
        root_seed (bytes): 用于生成随机数的种子。
        committee_size (int): 委员会中成员的数量。
        num_clients (int): 总的客户端数量。

    Returns:
        set: 包含委员会成员 ID 的集合。
    """

    # 使用 ChaCha20 密码算法，以 root_seed 为密钥，nonce 为常量，生成随机字节流。
    prg_committee_holder = ChaCha20.new(key=root_seed, nonce=nonce)

    # 生成 committee_size * 128 个字节的随机数据。
    data = b"secr" * committee_size * 128
    # 使用 prg_committee_holder 对随机数据进行加密，得到随机字节流。
    prg_committee_bytes = prg_committee_holder.encrypt(data)
    # 将加密后的字节流解析为一个数组，数据类型为 vector_type。
    committee_numbers = np.frombuffer(prg_committee_bytes, dtype=vector_type)

    # 初始化一个空集合，用于存储委员会成员 ID。
    user_committee = set()
    # 初始化计数器，用于遍历随机数组。
    cnt = 0

    # 循环直到委员会成员数量达到 committee_size。
    while (len(user_committee) < committee_size):
        # 从随机数组中获取一个元素，并对其进行模 num_clients 操作，得到一个在 0 到 num_clients 之间的随机整数。
        sampled_id = committee_numbers[cnt] % num_clients
        # 将随机整数添加到委员会成员集合中。
        (user_committee).add(sampled_id)
        # 计数器加 1。
        cnt += 1

    # 返回包含委员会成员 ID 的集合。
    return user_committee

# choose neighbors
def findNeighbors(root_seed, current_iteration, num_clients, id, neighborhood_size):
    neighbors_list = set() # a set, instead of a list

    # compute PRF(root, iter_num), output a seed. can use AES
    prf = ChaCha20.new(key=root_seed, nonce=nonce)
    current_seed = prf.encrypt(current_iteration.to_bytes(32, 'big')) 
  
    # compute PRG(seed), a binary string
    prg = ChaCha20.new(key=current_seed, nonce=nonce)
   
    # compute number of bytes we need for a graph
    num_choose = math.ceil(math.log2(num_clients))  # number of neighbors I choose
    num_choose = num_choose * neighborhood_size

    bytes_per_client = math.ceil(math.log2(num_clients) / 8)
    segment_len = num_choose * bytes_per_client
    num_rand_bytes = segment_len * num_clients
    data = b"a" * num_rand_bytes
    graph_string = prg.encrypt(data)

       
    # find the segment for myself
    my_segment = graph_string[id * segment_len: (id + 1) * segment_len]

    # define the number of bits within bytes_per_client that can be convert to int (neighbor's ID)
    bits_per_client = math.ceil(math.log2(num_clients))
    # default number of clients is power of two
    for i in range(num_choose):
        tmp = my_segment[i * bytes_per_client: (i + 1) * bytes_per_client]
        tmp_neighbor = int.from_bytes(tmp, 'big') & ((1 << bits_per_client)-1)
            
        if tmp_neighbor == id: # random neighbor choice happened to be itself, skip 
            continue
        if tmp_neighbor in neighbors_list: # client already chose tmp_neighbor, skip
            continue
        neighbors_list.add(tmp_neighbor)

    # now we have a list for who I chose
    # find my ID in the rest, see which segment I am in. add to neighbors_list
    for i in range(num_clients):
        if i == id:
            continue
        seg = graph_string[i * segment_len: (i + 1) * segment_len]
        ls = parse_segment_to_list(seg, num_choose, bits_per_client, bytes_per_client)
        if id in ls:
            neighbors_list.add(i)  # add current segment owner into neighbors_list
    
    return neighbors_list

def parse_segment_to_list(segment, num_choose, bits_per_client, bytes_per_client):
    cur_ls = set()
    # take a segment (byte string), parse it to a list
    for i in range(num_choose):
        cur_bytes = segment[i * bytes_per_client: (i + 1) * bytes_per_client]     
        cur_no = int.from_bytes(cur_bytes, 'big') & ((1 << bits_per_client) - 1)
        cur_ls.add(cur_no)
        
    return cur_ls

