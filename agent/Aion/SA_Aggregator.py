import time
import logging
import re
import dill

import pandas as pd
import numpy as np
import sympy
import random

from agent.Agent import Agent
from message.Message import Message
from util import param, util
from util.crypto.secretsharing.vss import VSS

class SA_AggregatorAgent(Agent):
    """
    Represents a server agent participating in a secure aggregation protocol.
    """

    def __str__(self):
        return "[server]"

    def __init__(self, id, name, type,
                 random_state=None,
                 msg_fwd_delay=1_000_000,  # Delay for forwarding peer-to-peer client messages (in nanoseconds)
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
        Initializes the server agent.

        Args:
            id (int): Agent ID.
            name (str): Agent name.
            type (str): Agent type.
            random_state (numpy.random.RandomState): Random number generator.
            msg_fwd_delay (int): Time for forwarding peer-to-peer client messages.
            round_time (pandas.Timedelta): Waiting time per round.
            iterations (int): Number of iterations.
            key_length (int): Key length.
            num_clients (int): Number of users for each training round.
            parallel_mode (int): Parallel mode.
            debug_mode (int): Debug mode.
            Dimension (int): Vector length.
            commit_size (int): Committee size.
            msg_name (str): Message name.
            users (set): Set of user IDs.
        """
        super().__init__(id, name, type, random_state)

        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if debug_mode:
            logging.basicConfig(level=logging.DEBUG)

        # System parameters
        self.msg_fwd_delay = msg_fwd_delay
        self.round_time = round_time
        self.no_of_iterations = iterations
        self.parallel_mode = parallel_mode

        # Input parameters
        self.num_clients = num_clients
        self.users = users if users is not None else set()
        self.vector_len = Dimension
        self.vector_dtype = param.vector_type

        # Security parameters
        self.commit_size = commit_size
        self.msg_name = msg_name
        self.committee_threshold = 0
        self.prime = param.prime

        # Initialize VSS
        self.vss = VSS()

        # Data storage
        self.times = []
        self.client_id_list = None
        self.seed_sum_hprf = None
        self.selected_indices = None
        self.committee_shares_sum = None
        self.seed_sum = None
        self.recv_user_masked_vectors = {}
        self.recv_committee_shares_sum = {}
        self.user_masked_vectors = {}
        self.user_committee = {}
        self.committee_sigs = {}
        self.recv_committee_sigs = {}
        self.receive_mask = {}
        self.mask_commitments = {}  # Store commitments for verification

        # Initialize vectors
        self.vec_sum_partial = np.zeros(self.vector_len, dtype=self.vector_dtype)
        self.final_sum = np.zeros(self.vector_len, dtype=self.vector_dtype)

        # Track current protocol iteration and round
        self.current_iteration = 1
        self.current_round = 0

        # Parameters for poison defense module
        self.l2_old = []
        self.linf_old = 0.1
        self.linf_shprg_old = 0.05
        self.b_old = 0.2

        # Accumulated time
        self.elapsed_time = {
            'REPORT': pd.Timedelta(0),
            'CROSSCHECK': pd.Timedelta(0),
            'RECONSTRUCTION': pd.Timedelta(0),
        }

        # Message processing map
        self.aggProcessingMap = {
            0: self.initialize,
            1: self.report,
            2: self.forward_signatures,
            3: self.reconstruction,
        }

        # Round name map
        self.namedict = {
            0: "initialize",
            1: "report",
            2: "forward_signatures",
            3: "reconstruction",
        }

        # Record time for each part
        self.timings = {
            "Legal clients confirmation": [],
            "Masked model generation": [],
            "Online clients confirmation": [],
            "Aggregate share reconstruction": [],
            "Model aggregation": [],
        }

    # Simulate lifecycle message
    def kernelStarting(self, startTime):
        """
        Initializes the server state when the kernel starts.

        Args:
            startTime (pandas.Timestamp): Kernel start time.
        """
        self.starttime = time.time()
        self.setComputationDelay(0)
        self.kernel.custom_state['Legal clients confirmation'] = 0
        self.kernel.custom_state['Masked model generation'] = 0
        self.kernel.custom_state['Online clients confirmation'] = 0
        self.kernel.custom_state['Aggregate share reconstruction'] = 0
        self.kernel.custom_state['Model aggregation'] = 0
        
        # 动态导入 ClientAgent 以避免循环导入
        from agent.Aion.SA_ClientAgent import SA_ClientAgent as ClientAgent
        self.ClientAgent = ClientAgent
        
        super().kernelStarting(startTime)

    def kernelStopping(self):
        """
        Performs cleanup when the kernel stops, including calculating average times.
        """
        # Calculate average times and record in the kernel state


        # Print total time and time for each part
        self.stoptime = time.time()
        # print("Time statistics for each part:")
        # for part, times in self.timings.items():
        #     if times:
        #         avg_time = sum(times) / len(times)
        #         print(f"{part}: Average time {avg_time:.6f} seconds, Total time {sum(times):.6f} seconds")
        self.kernel.custom_state['Legal clients confirmation'] = sum(self.timings["Legal clients confirmation"]) / len(self.timings["Legal clients confirmation"])
        self.kernel.custom_state['Masked model generation'] = sum(self.timings["Masked model generation"]) / len(self.timings["Masked model generation"])
        self.kernel.custom_state['Online clients confirmation'] = sum(self.timings["Online clients confirmation"]) / len(self.timings["Online clients confirmation"])
        self.kernel.custom_state['Aggregate share reconstruction'] = sum(self.timings["Aggregate share reconstruction"]) / len(self.timings["Aggregate share reconstruction"])
        self.kernel.custom_state['Model aggregation'] = sum(self.timings["Model aggregation"])  / len(self.timings["Model aggregation"])

        super().kernelStopping()

    def wakeup(self, currentTime):
        """
        Called at the end of each round, performs processing according to the current round.

        Args:
            currentTime (pandas.Timestamp): Current simulation time.
        """
        super().wakeup(currentTime)
        self.agent_print(
            f"wakeup in iteration {self.current_iteration} at function {self.namedict[self.current_round]}; current time is {currentTime}")
        self.aggProcessingMap[self.current_round](currentTime)

    def receiveMessage(self, currentTime, msg):
        """
        Receives and stores messages.

        Args:
            currentTime (pandas.Timestamp): Current time.
            msg (Message): Received message.
        """
        super().receiveMessage(currentTime, msg)
        sender_id = msg.body['sender']

        if msg.body['msg'] == "VECTOR" and msg.body['iteration'] == self.current_iteration:
            self.recv_user_masked_vectors[sender_id] = msg.body['masked_vector']

        elif msg.body['msg'] == "SIGN" and msg.body['iteration'] == self.current_iteration:
            self.recv_committee_sigs[sender_id] = msg.body['signed_labels']

        elif msg.body['msg'] == "hprf_SUM_SHARES" and msg.body['iteration'] == self.current_iteration:
            self.recv_committee_shares_sum[sender_id] = msg.body['sum_shares']

        elif msg.body['msg'] == "BFT_SIGN" and msg.body['iteration'] == self.current_iteration:
            self.recv_committee_sigs[sender_id] = msg.body['signed_labels']

    def initialize(self, currentTime):
        """
        Initializes the protocol, including selecting committee members and sending initial models.

        Args:
            currentTime (pandas.Timestamp): Current simulation time.
        """
        start_time = time.time()
        dt_protocol_start = pd.Timestamp('now')
        self.user_committee = param.choose_committee(param.root_seed, self.commit_size, self.num_clients)
        self.committee_threshold = len(self.user_committee) // 4

        initial_model_weights = np.ones(self.vector_len, dtype=self.vector_dtype) * 1000
        message = Message({"msg": "INITIAL_MODEL", "iteration": 0, "model_weights": initial_model_weights})


        self.client_id_list = [i for i in range(self.num_clients)]
        start_time = time.time()
        self.recv_committee_sigs = self.BFT_broadcast(message, self.client_id_list)
        self.timings["Legal clients confirmation"].append(time.time() - start_time)

        self.current_round = 1
        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        self.setWakeup(currentTime + server_comp_delay + pd.Timedelta('2s'))

    def BFT_broadcast(self, message, client_ids):
        """
        Broadcasts a message to the specified clients and waits for a response.

        Args:
            message (Message): Message to broadcast.
            client_ids (list): List of target client IDs.
        Returns:
            dict: Returns the received signed messages.
        """
        for client_id in client_ids:
            self.sendMessage(client_id, message, tag="comm_dec_server", msg_name=self.msg_name)

        start_time = time.time()
        Delta = pd.Timedelta('0.005s')
        while time.time() - start_time < 2 * Delta.total_seconds():
            if len(self.recv_committee_sigs) == len(client_ids):
                break
            time.sleep(0.0001)

        return self.recv_committee_sigs

    def report(self, currentTime):
        """
        Handles masked vectors from clients and sends decryption requests.

        Args:
            currentTime (pandas.Timestamp): Current simulation time.
        """
        start_time = time.time()
        self.report_time = time.time()
        dt_protocol_start = pd.Timestamp('now')

        self.report_read_from_pool()
        self.report_process()
        self.report_clear_pool()

        # BFT confirmation of online client information
        online_clients_list = [1 if i in self.recv_user_masked_vectors else 0 for i in range(self.num_clients)]
        online_clients_list_bytes = bytes(online_clients_list)
        message_online_clients = Message({"msg": "ONLINE_CLIENTS", "iteration": self.current_iteration,
                                          "online_clients": online_clients_list_bytes})
        self.recv_committee_sigs = self.BFT_broadcast(message_online_clients, self.user_committee)

        self.report_send_message()

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        self.agent_print("Report step running time:", server_comp_delay)

        self.recordTime(dt_protocol_start, "REPORT")
        self.report_time = time.time() - self.report_time

        self.current_round = 2
        self.setWakeup(currentTime + server_comp_delay)
        self.timings["Masked model generation"].append(time.time() - start_time)

    def report_read_from_pool(self):
        """Reads data from the receiving pool."""
        self.user_masked_vectors = self.recv_user_masked_vectors
        self.recv_user_masked_vectors = {}

    def report_clear_pool(self):
        """Clears the message pool."""
        self.recv_committee_shares_sum = {}
        self.recv_committee_sigs = {}

    def report_process(self):
        """
        Processes masked vectors and calculates the partial sum.
        """
        self.agent_print("Number of collected vectors:", len(self.user_masked_vectors))

        self.client_id_list = list(self.user_masked_vectors.keys())

        self.selected_indices, self.b_old = self.MMF(self.user_masked_vectors, self.l2_old, self.linf_old,
                                                     self.linf_shprg_old, self.b_old,
                                                     self.current_iteration)

        self.vec_sum_partial = np.zeros(self.vector_len, np.int64)
        for id in self.selected_indices:
            if len(self.user_masked_vectors[id]) != self.vector_len:
                raise RuntimeError("Client sent a vector with an incorrect length.")
            self.vec_sum_partial += self.user_masked_vectors[id]
            self.vec_sum_partial %= self.prime

    def report_send_message(self):
        """Sends decryption requests to committee members."""
        for id in self.user_committee:
            self.sendMessage(id,
                             Message({"msg": "SIGN",
                                      "iteration": self.current_iteration,
                                      "client_id_list": self.client_id_list,
                                      }),
                             tag="comm_dec_server",
                             msg_name=self.msg_name)

    def forward_signatures(self, currentTime):
        """
        Forwards signatures and requests secret shares.

        Args:
            currentTime (pandas.Timestamp): Current simulation time.
        """
        dt_protocol_start = pd.Timestamp('now')
        self.check_time = time.time()

        for id in self.user_committee:
            self.sendMessage(id,
                             Message({"msg": "request shares sum",
                                      "iteration": self.current_iteration,
                                      "request id list": self.selected_indices,
                                      }),
                             tag="comm_sign_server",
                             msg_name=self.msg_name)

        self.current_round = 3
        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        self.agent_print("Crosscheck step running time:", server_comp_delay)
        self.setWakeup(currentTime + server_comp_delay + param.wt_Aion_reconstruction)

        self.recordTime(dt_protocol_start, "CROSSCHECK")
        self.check_time = time.time() - self.check_time

    def reconstruction(self, currentTime):
        """
        Performs vector reconstruction and sends the final result to clients.

        Args:
            currentTime (pandas.Timestamp): Current simulation time.
        """
        dt_protocol_start = pd.Timestamp('now')
        self.reco_time = time.time()

        self.reconstruction_read_from_pool()
        self.reconstruction_process()
        self.reconstruction_clear_pool()
        self.reconstruction_send_message()

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        self.agent_print("Reconstruction time:", server_comp_delay)
        self.recordTime(dt_protocol_start, "RECONSTRUCTION")
        self.reco_time = time.time() - self.reco_time

        print()
        print("######## Iteration completed ########")
        print(f"[Server] Completed iteration {self.current_iteration} at {currentTime + server_comp_delay}")
        print()

        self.current_round = 1
        self.current_iteration += 1
        if self.current_iteration > self.no_of_iterations:
            return

        self.setWakeup(currentTime + server_comp_delay + param.wt_Aion_report)

    def reconstruction_read_from_pool(self):
        """Reads decryption shares from the receiving pool."""
        while len(self.recv_committee_shares_sum) < self.committee_threshold:
            time.sleep(0.01)

        self.committee_shares_sum = self.recv_committee_shares_sum
        self.recv_committee_shares_sum = {}

    def reconstruction_clear_pool(self):
        """Clears all message pools."""
        self.user_masked_vectors = {}
        self.committee_shares_sum = {}

        self.recv_user_masked_vectors = {}
        self.recv_committee_shares_sum = {}
        self.recv_user_masked_vectors = {}

    def reconstruction_process(self):
        """Processes secret shares, reconstructs the mask, and the final vector sum."""
        self.agent_print("Number of collected shares:", len(self.committee_shares_sum))
        if len(self.committee_shares_sum) < self.committee_threshold:
            raise RuntimeError("Not enough decryption shares received.")

        start_time = time.time()
        committee_shares_sum_list = [shares_list for _, shares_list in self.committee_shares_sum.items()]
        self.seed_sum_hprf = SA_AggregatorAgent.reconstruct_secret_vector(committee_shares_sum_list, self.prime)
        self.timings["Aggregate share reconstruction"].append(time.time() - start_time)

        start_time = time.time()
        self.final_sum = self.vec_sum_partial - self.seed_sum_hprf
        self.final_sum //= len(self.selected_indices)
        self.final_sum = np.array(self.final_sum, dtype=np.uint32)
        self.final_sum %= self.prime

        self.l2_old = [np.linalg.norm(self.final_sum)] + self.l2_old[:1]
        self.linf_old = np.max(np.abs(self.final_sum))
        self.linf_shprg_old = np.max(np.abs(self.seed_sum_hprf))
        message_final_sum = Message(
            {"msg": "FINAL_SUM", "iteration": self.current_iteration, "final_sum": self.final_sum})

        self.timings["Model aggregation"].append(time.time() - start_time)
        start_time = time.time()
        self.BFT_broadcast(message_final_sum, self.user_committee)
        self.timings["Online clients confirmation"].append(time.time() - start_time)

    @staticmethod
    def reconstruct_secret(shares: list, prime: int) -> int:
        """
        Recovers the secret value from secret shares.

        Args:
            shares (list): List of secret shares.
            prime (int): The prime number used.

        Returns:
            int: The recovered secret value.
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
        Recovers a secret vector from secret shares, with all elements sharing the same Lagrangian coefficients.

        Args:
            shares (list): List of secret shares for each vector element.
            prime (int): The prime number used.

        Returns:
            list: The recovered secret vector.
        """
        n = len(shares)  # Dimension of the vector
        k = len(shares[0])  # Number of shares used for reconstruction (assuming all elements have same number of shares)

        # Pre-compute Lagrange coefficients since they are the same for all elements
        lagrange_coefficients = []
        for i in range(n):
            numerator, denominator = 1, 1
            x_i, _ = shares[i][0]  # Get x value of the first share since x is same for all elements
            for j in range(n):
                if i == j:
                    continue
                x_j, _ = shares[j][0]
                numerator = (numerator * (-1 * x_j)) % prime
                denominator = (denominator * (x_i - x_j)) % prime
            lagrange_coefficients.append((numerator * sympy.mod_inverse(denominator, prime)) % prime)

        secret_vector = []
        for i in range(k):
            secret_element = 0
            for j in range(n):
                _, y_i = shares[j][i]
                secret_element = (secret_element + y_i * lagrange_coefficients[j]) % prime  # Using the pre-computed Lagrange coefficients
            secret_vector.append(secret_element % prime)
        return secret_vector

    @staticmethod
    def vss_reconstruct(shares: list, prime: int) -> int:
        """
        Local function to recover the secret.

        Args:
            shares (list): List of secret shares.
            prime (int): The prime number used.

        Returns:
            int: The recovered secret.
        """
        return SA_AggregatorAgent.reconstruct_secret(shares, prime)

    def reconstruction_send_message(self):
        """Sends the final result to clients."""
        for id in self.users:
            self.sendMessage(id,
                             Message({"msg": "REQ", "sender": 0, "output": 1}),
                             tag="comm_output_server",
                             msg_name=self.msg_name)

    def MMF(self, masked_updates, l2_old, linf_old, linf_shprg_old, b_old, current_round):
        """
        Function to select benign clients.

        Args:
            masked_updates (dict): Dictionary of masked updates from clients, where the key is the client index (int) and value is a 1D numpy array.
            ... (other parameters remain unchanged)

        Returns:
            list: List of benign client indices.
            float: Threshold b for the current round.
        """
        WEIGHT = 1.0  # Weight
        MIN_THRESHOLD = 0.3  # Minimum threshold
        RESUME = False
        RESUMED_NAME = None

        cnt = len(masked_updates)

        # Calculate L2 norm and sort, keep original index
        l2_norm = {k: np.linalg.norm(v) for k, v in masked_updates.items()}
        sorted_l2_norm = dict(sorted(l2_norm.items(), key=lambda item: item[1]))

         # Dynamically adjust threshold b
        if current_round <= 3 or (
                RESUME and current_round <= int(
                    re.findall(r'\d+\d*', RESUMED_NAME.split('/')[1])[0]) + 3 if RESUMED_NAME else 0):
            b = list(sorted_l2_norm.values())[int(MIN_THRESHOLD * cnt)]
        else:
            b = (l2_old[1] + linf_shprg_old) / (l2_old[0] + linf_shprg_old) * b_old

        # Select benign clients using the sorted dictionary
        selected_indices = []
        count = 0
        for k, v in sorted_l2_norm.items():
            if v <= b:
                selected_indices.append(k)
                count += 1
            if count >= int(0.8 * cnt):  # Limit maximum number here
                break

        benign_index = max(int(MIN_THRESHOLD * cnt),
                           min(int(0.8 * cnt), len(selected_indices)))
        if len(selected_indices) > benign_index:
            selected_indices = selected_indices[:benign_index]
        else:
            selected_indices = list(sorted_l2_norm.keys())[:benign_index]
        return selected_indices, b

    def vss_verify_share(self, share, commitments, prime=None):
        """
        Verify a share against the commitments.
        
        Args:
            share: A share in the format (share_index, share_value).
            commitments: List of commitments.
            prime: The prime number to use.
            
        Returns:
            is_valid: True if the share is valid, False otherwise.
        """
        if prime is None:
            prime = self.prime
            
        return self.vss.verify_share(share, commitments, prime)
        
    def vss_reconstruct(self, shares, prime=None):
        """
        Reconstruct the secret from shares.
        
        Args:
            shares: List of shares in the format [(share_index, share_value)].
            prime: The prime number to use.
            
        Returns:
            secret: The reconstructed secret.
        """
        if prime is None:
            prime = self.prime
            
        return self.vss.reconstruct(shares, prime)

    # ======================== UTIL ========================
    def recordTime(self, startTime, categoryName):
        """
        Records the elapsed time.

        Args:
            startTime (pandas.Timestamp): Start time.
            categoryName (str): Category name.
        """
        dt_protocol_end = pd.Timestamp('now')
        self.elapsed_time[categoryName] += dt_protocol_end - startTime

    def agent_print(*args, **kwargs):
        """
        Custom print function that adds a [Server] header before printing.
        """
        print(f"[Server] ", *args, **kwargs)

    def handleMessage(self, currentTime, msg):
        """
        Handles incoming messages.

        Args:
            currentTime (pandas.Timestamp): The current simulation time.
            msg (Message): The message to handle.
        """
        if msg.body["msg"] == "VECTOR":
            self.handle_vector(currentTime, msg)
        elif msg.body["msg"] == "SHARED_MASK":
            self.handle_shared_mask(currentTime, msg)
        elif msg.body["msg"] == "MASK_COMMITMENTS":
            self.handle_mask_commitments(currentTime, msg)
        elif msg.body["msg"] == "SIGN":
            self.handle_sign(currentTime, msg)
        elif msg.body["msg"] == "SHARES_SUM":
            self.handle_shares_sum(currentTime, msg)
        elif msg.body["msg"] == "FINAL_RESULT":
            self.handle_final_result(currentTime, msg)
        else:
            self.agent_print(f"Unknown message type: {msg.body['msg']}")

    def handle_shared_mask(self, currentTime, msg):
        """
        Handles a shared mask message.

        Args:
            currentTime (pandas.Timestamp): The current simulation time.
            msg (Message): The message containing the shared mask.
        """
        sender_id = msg.body["sender"]
        shared_mask = msg.body["shared_mask"]
        commitments = msg.body.get("commitments", None)
        
        # Verify the share if commitments are provided
        if commitments is not None:
            is_valid = self.vss_verify_share(shared_mask, commitments, self.prime)
            if not is_valid:
                self.agent_print(f"Invalid share received from client {sender_id}")
                return
        
        self.recv_shared_masks[sender_id] = shared_mask
        self.agent_print(f"Received shared mask from client {sender_id}")
        
        # Check if we have received all shared masks
        if len(self.recv_shared_masks) == len(self.selected_indices):
            self.agent_print("Received all shared masks")
            self.crosscheck(currentTime)
            
    def handle_mask_commitments(self, currentTime, msg):
        """
        Handles mask commitments message.

        Args:
            currentTime (pandas.Timestamp): The current simulation time.
            msg (Message): The message containing the mask commitments.
        """
        sender_id = msg.body["sender"]
        commitments = msg.body["commitments"]
        
        # Store the commitments for verification
        self.mask_commitments[sender_id] = commitments
        self.agent_print(f"Received mask commitments from client {sender_id}")
        
        # Check if we have received all commitments
        if len(self.mask_commitments) == len(self.selected_indices):
            self.agent_print("Received all mask commitments")
            
    def verify_all_shares(self):
        """
        Verifies all shares against their commitments.
        
        Returns:
            is_valid: True if all shares are valid, False otherwise.
        """
        for client_id, shares in self.recv_shared_masks.items():
            if client_id in self.mask_commitments:
                commitments = self.mask_commitments[client_id]
                is_valid = self.vss_verify_share(shares, commitments, self.prime)
                if not is_valid:
                    self.agent_print(f"Invalid share from client {client_id}")
                    return False
        return True

    def crosscheck(self, currentTime):
        """
        Performs crosscheck and requests shares from committee members.

        Args:
            currentTime (pandas.Timestamp): Current simulation time.
        """
        dt_protocol_start = pd.Timestamp('now')
        self.check_time = time.time()
        
        # Verify all shares
        if not self.verify_all_shares():
            self.agent_print("Some shares are invalid, aborting crosscheck")
            return

        for id in self.user_committee:
            self.sendMessage(id,
                             Message({"msg": "request shares sum",
                                      "iteration": self.current_iteration,
                                      "request id list": self.selected_indices,
                                      }),
                             tag="comm_sign_server",
                             msg_name=self.msg_name)

        self.current_round = 3
        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        self.agent_print("Crosscheck step running time:", server_comp_delay)
        self.setWakeup(currentTime + server_comp_delay + param.wt_Aion_reconstruction)

        self.recordTime(dt_protocol_start, "CROSSCHECK")
        self.check_time = time.time() - self.check_time