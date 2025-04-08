# -*- coding: utf-8 -*-
"""
    Verifiable Secret Sharing (VSS)
    ~~~~~

    Implementation of Feldman's Verifiable Secret Sharing scheme.
    Based on the paper: "A Practical Scheme for Non-interactive Verifiable Secret Sharing"
    by Paul Feldman (1987).
"""

import random
from Cryptodome.Hash import SHA256
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
from .polynomials import random_polynomial, get_polynomial_points, mod_inverse
from .primes import get_large_enough_prime


class VSS:
    """
    Implements Feldman's Verifiable Secret Sharing scheme.
    """
    
    def __init__(self, g=None, p=None, q=None):
        """
        Initialize the VSS scheme with generator g and prime p.
        
        Args:
            g: Generator for the cyclic group. If None, a default value will be used.
            p: Prime number for the field. If None, a suitable prime will be generated.
            q: Prime number for the order of the cyclic group. If None, p-1 will be used.
        """
        # Use default values for demonstration
        # In a real implementation, these should be carefully chosen
        if p is None:
            p = 2**256 - 2**32 - 977  # A 256-bit prime
        if q is None:
            q = (p - 1) // 2  # A 255-bit prime
        if g is None:
            g = 2  # Generator for the cyclic group
            
        self.p = p
        self.q = q
        self.g = g
        
    def _hash(self, data):
        """Hash data using SHA-256."""
        h = SHA256.new()
        h.update(data)
        return bytes_to_long(h.digest()) % self.p
    
    def _commitment(self, coefficients):
        """
        Generate commitments for the polynomial coefficients.
        
        Args:
            coefficients: List of polynomial coefficients.
            
        Returns:
            commitments: List of commitments for each coefficient.
        """
        commitments = []
        for coeff in coefficients:
            # Compute g^coeff mod p
            commitment = pow(self.g, coeff, self.p)
            commitments.append(commitment)
        return commitments
    
    def share(self, secret, num_shares, threshold, prime=None):
        """
        Share a secret using VSS.
        
        Args:
            secret: The secret to be shared.
            num_shares: Number of shares to generate.
            threshold: Number of shares required to reconstruct the secret.
            prime: Prime number for the field. If None, a suitable prime will be generated.
            
        Returns:
            shares: List of shares in the format [(share_index, share_value)].
            commitments: List of commitments for verification.
        """
        if prime is None:
            prime = get_large_enough_prime([secret, num_shares])
            if prime is None:
                raise ValueError("Error! Secret is too long for share calculation!")
        
        # Generate a random polynomial with the secret as the constant term
        coefficients = random_polynomial(threshold-1, secret, prime)
        
        # Generate shares
        shares = get_polynomial_points(coefficients, num_shares, prime)
        
        # Generate commitments
        commitments = self._commitment(coefficients)
        
        return shares, commitments
    
    def verify_share(self, share, commitments, prime):
        """
        Verify a share against the commitments.
        
        Args:
            share: A share in the format (share_index, share_value).
            commitments: List of commitments.
            prime: Prime number for the field.
            
        Returns:
            is_valid: True if the share is valid, False otherwise.
        """
        i, share_value = share
        
        # Compute g^share_value mod p
        left_side = pow(self.g, share_value, self.p)
        
        # Compute the product of commitments raised to powers of i
        right_side = 1
        for j, commitment in enumerate(commitments):
            right_side = (right_side * pow(commitment, i**j, self.p)) % self.p
        
        return left_side == right_side
    
    def reconstruct(self, shares, prime):
        """
        Reconstruct the secret from shares.
        
        Args:
            shares: List of shares in the format [(share_index, share_value)].
            prime: Prime number for the field.
            
        Returns:
            secret: The reconstructed secret.
        """
        # Use Lagrange interpolation to reconstruct the secret
        # The secret is the constant term of the polynomial
        x_values, y_values = zip(*shares)
        
        # Simple Lagrange interpolation
        secret = 0
        for i, (x_i, y_i) in enumerate(shares):
            numerator = 1
            denominator = 1
            for j, (x_j, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-x_j)) % prime
                    denominator = (denominator * (x_i - x_j)) % prime
            
            # Compute the Lagrange coefficient
            lagrange_coeff = (numerator * mod_inverse(denominator, prime)) % prime
            
            # Add the term to the secret
            secret = (secret + (y_i * lagrange_coeff) % prime) % prime
        
        return secret


# Example usage
if __name__ == "__main__":
    # Create a VSS instance
    vss = VSS()
    
    # Share a secret
    secret = 42
    num_shares = 5
    threshold = 3
    prime = 2**32 - 5  # A 32-bit prime
    
    shares, commitments = vss.share(secret, num_shares, threshold, prime)
    
    # Verify shares
    for share in shares:
        is_valid = vss.verify_share(share, commitments, prime)
        print(f"Share {share} is valid: {is_valid}")
    
    # Reconstruct the secret
    reconstructed_secret = vss.reconstruct(shares[:threshold], prime)
    print(f"Original secret: {secret}")
    print(f"Reconstructed secret: {reconstructed_secret}")
    print(f"Reconstruction successful: {secret == reconstructed_secret}") 