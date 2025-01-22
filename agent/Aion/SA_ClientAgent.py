import torch

from agent.Agent import Agent
from agent.HPRF.hprf import load_initialization_values, SHPRG
from agent.Aion.SA_ServiceAgent import SA_ServiceAgent as ServiceAgent
from message.Message import Message
import dill
import time
import logging
import pandas as pd
import random
from Cryptodome.Hash import SHA256
from Cryptodome.Signature import DSS

from util import param
from util import util

from agent.Aion.tool import *


class SA_ClientAgent(Agent):
    """Represents a client agent participating in a secure aggregation protocol."""

    def __str__(self):
        return "[client]"

    def __init__(self, id, name, type,
                 iterations=4,
                 key_length=32,
                 num_clients=128,
                 neighborhood_size=1,
                 debug_mode=0,
                 Dimension=10000,
                 commit_size=8,
                 msg_name=None,
                 random_state=None):
        """
        Initializes the client agent.

        Args:
            id (int): Unique ID of the agent.
            name (str): Name of the agent.
            type (str): Type of the agent.
            iterations (int, optional): Number of iterations for the protocol. Defaults to 4.
            key_length (int, optional): Length of the encryption key in bytes. Defaults to 32.
            num_clients (int, optional): Number of clients participating in the protocol. Defaults to 128.
            neighborhood_size (int, optional): Number of neighbors for each client. Defaults to 1.
            debug_mode (int, optional): Whether to enable debug mode. Defaults to 0.
            random_state (random.Random, optional): Random number generator. Defaults to None.
        """

        super().__init__(id, name, type, random_state)

        self.commit_size = commit_size
        self.msg_name = msg_name
        self.report_time = None
        self.reco_time = None
        self.check_time = None
        self.cipher_stored = None
        self.key_length = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if debug_mode:
            logging.basicConfig()

        self.key = util.read_key(f"pki_files/client{self.id}.pem")


        self.num_clients = num_clients
        self.neighborhood_size = neighborhood_size
        self.vector_len = Dimension
        self.vector_dtype = param.vector_type

        self.key_length = key_length

        self.user_committee = param.choose_committee(param.root_seed,
                                                     self.commit_size,
                                                     self.num_clients)

        self.committee_shared_sk = None
        self.committee_member_idx = None

        self.prime = param.prime

        self.elapsed_time = {'REPORT': pd.Timedelta(0),
                             'CROSSCHECK': pd.Timedelta(0),
                             'RECONSTRUCTION': pd.Timedelta(0),
                             }

        self.initial_time = 0
        self.ag_time = 0

        self.no_of_iterations = iterations
        self.current_iteration = 1
        self.current_base = 0

        self.setup_complete = False
        self.mask_seeds = []
        self.receive_mask_shares = [0 for _ in range(5000)]

    def kernelStarting(self, startTime):
        """
        Called when the simulation starts.

        Args:
            startTime (pandas.Timestamp): The start time of the simulation.
        """
        if self.id == 0:
            self.kernel.custom_state['clt_report'] = pd.Timedelta(0)
            self.kernel.custom_state['clt_crosscheck'] = pd.Timedelta(0)
            self.kernel.custom_state['clt_reconstruction'] = pd.Timedelta(0)

        self.serviceAgentID = self.kernel.findAgentByType(ServiceAgent)

        self.setComputationDelay(0)

        super().kernelStarting(startTime +
                               pd.Timedelta(self.random_state.randint(low=0, high=1000), unit='ns'))

    def kernelStopping(self):
        """
        Called when the simulation stops.
        """

        self.kernel.custom_state['clt_report'] += (
                self.elapsed_time['REPORT'] / self.no_of_iterations)
        self.kernel.custom_state['clt_crosscheck'] += (
                self.elapsed_time['CROSSCHECK'] / self.no_of_iterations)
        self.kernel.custom_state['clt_reconstruction'] += (
                self.elapsed_time['RECONSTRUCTION'] / self.no_of_iterations)

        super().kernelStopping()

    def wakeup(self, currentTime):
        """
        Called when the agent is awakened.

        Args:
            currentTime (pandas.Timestamp): The current simulation time.
        """

        self.report_time = time.time()
        super().wakeup(currentTime)
        dt_wake_start = pd.Timestamp('now')
        self.sendVectors(currentTime)
        self.report_time = time.time() - self.report_time


    def BFT_report(self, sign_message):
        """
        Signs a message and sends it to the server.

        Args:
            sign_message (Message): The message to be signed.
        """

        msg_to_sign = dill.dumps(sign_message.body)
        hash_container = SHA256.new(msg_to_sign)
        signer = DSS.new(self.key, 'fips-186-3')
        signature = signer.sign(hash_container)
        client_signed_labels = (hash_container.hexdigest(), signature)

        dt_protocol_start = pd.Timestamp('now')

        self.sendMessage(self.serviceAgentID,
                         Message({"msg": "BFT_SIGN",
                                  "iteration": self.current_iteration,
                                  "sender": self.id,
                                  "signed_labels": client_signed_labels,
                                  "sign_message": sign_message,
                                  }),
                         tag="comm_sign_client",
                         msg_name=self.msg_name)

        self.recordTime(dt_protocol_start, "CROSSCHECK")
        msg_to_sign = None
        hash_container = None
        signer = None
        signature = None
        client_signed_labels = None

        return client_signed_labels

    def receiveMessage(self, currentTime, msg):
        """
        Called when the agent receives a message.

        Args:
            currentTime (pandas.Timestamp): The current simulation time.
            msg (Message): The received message.
        """
        super().receiveMessage(currentTime, msg)

        if msg.body['msg'] == "request shares sum":
            if msg.body['iteration'] == self.current_iteration:
                dt_protocol_start = pd.Timestamp('now')
                self.reco_time = time.time()
                sum_shares = self.get_sum_shares(msg.body['request id list'])
                initialization_values_filename = r"agent\\HPRF\\initialization_values"
                n, m, p, q = load_initialization_values(initialization_values_filename)
                filename = r"agent\\HPRF\\matrix"
                shprg = SHPRG(n, m, p, q, filename)
                hprf_sum_shares = shprg.list_hprf(sum_shares, self.current_iteration, self.vector_len)
                clt_comp_delay = pd.Timestamp('now') - dt_protocol_start

                self.sendMessage(self.serviceAgentID,
                                 Message({"msg": "hprf_SUM_SHARES",
                                          "iteration": self.current_iteration,
                                          "sender": self.id,
                                          "sum_shares": hprf_sum_shares,
                                          }),
                                 tag="comm_secret_sharing",
                                 msg_name=self.msg_name)

                self.recordTime(dt_protocol_start, 'RECONSTRUCTION')
                self.reco_time = time.time() - self.reco_time
                self.recordTime(dt_protocol_start, 'RECONSTRUCTION')

        elif msg.body['msg'] == "REQ" and self.current_iteration != 0:
            self.current_iteration += 1
            if self.current_iteration > self.no_of_iterations:
                return

            dt_protocol_start = pd.Timestamp('now')
            self.sendVectors(currentTime)
            self.recordTime(dt_protocol_start, "REPORT")

        elif msg.body['msg'] == "ONLINE_CLIENTS" or msg.body['msg'] == "FINAL_SUM":
            if msg.body['iteration'] == self.current_iteration:
                self.BFT_report(msg)

        elif msg.body['msg'] == "SHARED_MASK":
            sender_id = msg.body['sender']
            temp_shared_mask = msg.body['shared_mask']
            self.receive_mask_shares[sender_id] = temp_shared_mask

    def sendVectors(self, currentTime):
        """
        Sends the vectors to the server.

        Args:
            currentTime (pandas.Timestamp): The current simulation time.
        """

        dt_protocol_start = pd.Timestamp('now')

        if self.current_iteration == 1:
            start_time = time.time()
            self.mask_seed = random.SystemRandom().randint(1, self.prime)
            self.share_mask_seed()
            self.initial_time = time.time() - start_time

        start_time = time.time()
        initialization_values_filename = r"agent\\HPRF\\initialization_values"
        n, m, p, q = load_initialization_values(initialization_values_filename)
        filename = r"agent\\HPRF\\matrix"
        shprg = SHPRG(n, m, p, q, filename)

        mask_vector = shprg.hprf(self.mask_seed, self.current_iteration, self.vector_len)
        mask_vector = np.array(mask_vector, dtype=np.uint32)

        vec = np.ones(self.vector_len, dtype=self.vector_dtype)

        masked_vec = vec + mask_vector

        self.sendMessage(self.serviceAgentID,
                         Message({"msg": "VECTOR",
                                  "iteration": self.current_iteration,
                                  "sender": self.id,
                                  "masked_vector": masked_vec,
                                  }),
                         tag="comm_key_generation",
                         msg_name=self.msg_name)
        self.ag_time = time.time() - start_time



    def share_mask_seed(self):
        """
        Generates and shares the mask seed.
        """
        shares = SA_ClientAgent.vss_share(self.mask_seed, len(self.user_committee),
                                              len(self.user_committee) - 4, self.prime)
        user_committee_list = list(self.user_committee)

        for j in range(len(user_committee_list)):
            self.sendMessage(user_committee_list[j],
                             Message({"msg": "SHARED_MASK",
                                      "sender": self.id,
                                      "shared_mask": shares[j],
                                      }),
                             tag="comm_secret_sharing",
                             msg_name=self.msg_name)
        pass

    def generate_shares(secret, num_shares, threshold, prime, seed=None):
        """
        Generates secret shares.

        Args:
            secret: The secret to be shared.
            num_shares: The number of shares to generate.
            threshold: The number of shares required to reconstruct the secret.
            prime: The prime number to use.
            seed: An optional seed for the random number generator.

        Returns:
            shares: A list of secret shares in the format [(share_index, share_value)].
        """
        if seed is not None:
            random.seed(seed)
        coefficients = [secret] + [random.SystemRandom().randrange(1, prime) for _ in range(threshold - 1)]
        polynomial = lambda x: sum([coeff * x ** i for i, coeff in enumerate(coefficients)])
        shares = [(x, polynomial(x) % prime) for x in range(1, num_shares + 1)]
        return shares

    def vss_share(secret, num_shares: int, threshold: int = None, prime=None, seed=None):
        """
        Local secret sharing function.

        Args:
            secret: The secret to be shared.
            num_shares: The number of shares to generate.
            threshold: The number of shares required to reconstruct the secret. Defaults to half of num_shares.
            prime: The prime number to use.
            seed: An optional seed for the random number generator.

        Returns:
            shares: A list of secret shares in the format [(share_index, share_value)].
        """
        if threshold is None:
            threshold = num_shares//2
        shares = SA_ClientAgent.generate_shares(secret, num_shares, threshold, prime, seed)
        return shares


    def sum_shares(shares_list, prime):
        """Sums multiple secret shares."""
        sum_shares = []
        sum_value = 0
        for share in shares_list:
            if share == 0:
                continue
            sum_value += share[1] % prime
        i = 0
        while 1:
            if shares_list[i] == 0:
                i += 1
                continue
            sum_shares.append((shares_list[i][0], sum_value))
            break
        return sum_shares

    def get_sum_shares(self,client_id_list):
        """
        Sums the secret shares.

        Args:
            client_id_list (list): List of client IDs.

        Returns:
            sum_shares: The sum of the secret shares.
        """
        dt_protocol_start = pd.Timestamp('now')

        shares = []
        for i in range(len(client_id_list)):
            shares.append(self.receive_mask_shares[client_id_list[i]])

        sum_shares = SA_ClientAgent.sum_shares(shares, self.prime)

        return sum_shares


    def recordTime(self, startTime, categoryName):
        """
        Records the time.

        Args:
            startTime (pandas.Timestamp): The start time.
            categoryName (str): The category name.
        """
        dt_protocol_end = pd.Timestamp('now')
        self.elapsed_time[categoryName] += dt_protocol_end - startTime

    def agent_print(*args, **kwargs):
        """
        Custom print function that adds a [Server] header before printing.

        Args:
            *args: Any positional arguments accepted by the built-in print function.
            **kwargs: Any keyword arguments accepted by the built-in print function.
        """
        print(*args, **kwargs)