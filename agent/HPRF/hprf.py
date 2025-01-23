import pickle
import os
import random
from decimal import Decimal, getcontext

getcontext().prec = 1024

class SHPRG:
    def __init__(self, n, m, p, q, filename):
        """
        Initializes the SHPRF object.

        Args:
            n (int): Number of rows in the matrix.
            m (int): Number of columns in the matrix.
            p (int): The first parameter for the modulo operation.
            q (int): The second parameter for the modulo operation.
            filename (str): The filename to store the matrix.
        """
        assert p < q, "p < q"  # Ensure p is less than q
        assert n < m, "n < m"  # Ensure n is less than m

        self.n = n
        self.m = m
        self.p = p
        self.q = q
        self.filename = filename

        # Load the matrix if the file exists
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                self.A = pickle.load(file)
        # Otherwise, generate a new random matrix and save it to the file
        else:
            self.A = [[random.randint(0, q - 1) for _ in range(m)] for _ in range(n)]
            with open(filename, 'wb') as file:
                pickle.dump(self.A, file)
    def G(self, s):
        """
        Calculates the matrix product A^T * s and maps the result to the range [0, p).

        Args:
            s (int): The seed value.

        Returns:
            list: The mapped result vector.
        """
        s = [[s]]
        product = []
        for j in range(self.m):
            sum_result = 0
            for i in range(self.n):
                sum_result += self.A[i][j] * s[i][0]
            product.append(sum_result % self.q)  # Modulo operation
        product = [Decimal(product[j]) for j in range(self.m)]
        p = Decimal(self.p)
        q = Decimal(self.q)
        result = [int((x * p / q + Decimal('0.5'))) for x in product]
        return result

    def hprf(self, k, x, length):
        result = []
        counter = 0
        while len(result) < length:
            s = (k * (x + counter)) % self.q
            result.extend(self.G(s))
            counter += 1
        return result[:length]


    def list_hprf(self, k, x, length):
        result = []
        counter = 0
        offset = k[0][0]
        initial_value = k[0][1]

        while len(result) < length:
            s = (initial_value * (x + counter)) % self.q
            generated_values = self.G(s)
            for val in generated_values:
                result.append((offset, val))

            counter += 1
        return result[:length]


    def hprg(self, seed, length):
        """
         Generates a pseudo-random sequence of a specified length.

         Args:
             seed (int): The seed value.
             length (int): The length of the sequence.

         Returns:
             list: The pseudo-random sequence.
         """
        s = seed
        extended_vector = self.G(s)
        if length > self.m:
            repeated_vector = (extended_vector * (length // self.m + 1))[:length]
            return repeated_vector
        else:
            return extended_vector[:length]


    def list_hprg(self, seed, length):
        """
        Generates a pseudo-random sequence of a specified length.

        Args:
            seed (int): The seed value.
            length (int): The length of the sequence.

        Returns:
            list: The pseudo-random sequence.
        """
        result = []
        offset = seed[0][0]
        initial_value = seed[0][1]
        extended_vector = self.G(initial_value)
        if length > self.m:
            repeated_vector = (extended_vector * (length // self.m + 1))[:length]
            for i in range(len(repeated_vector)):
                result.append((offset, repeated_vector[i]))
            return result
        else:
            for i in range(len(extended_vector)):
                result.append((offset, extended_vector[i]))
            return result[:length]


def load_initialization_values(filename):
    """
    Loads initialization values from a file.

    Args:
        filename (str): The filename.

    Returns:
        tuple: Initialization values (n, m, p, q).
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)