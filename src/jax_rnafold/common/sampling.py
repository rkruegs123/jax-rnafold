"""An interface to an efficient algorithm for uniformly sampling secondary structures for an RNA sequence."""

from jax_rnafold.common.utils import valid_pair, DEFAULT_HAIRPIN


class UniformStructureSampler:
    """ A class for sampling all possible structures for RNA primary sequences.
    For a given primary sequence, `precomp` runs an :math:`\mathcal{O}(n^3)`
    precomputation algorithm after which all structures are given an arbitrary
    unique ID. The structure for an ID can be retrieved in :math:`\mathcal{O}(n^2)`
    time. This is useful for generating a uniform random sample from the space of
    structures for a sequence.

    Attributes:
      hairpin: The minimum hairpin length. Defaults to 3.
    """

    def __init__(self, hairpin=DEFAULT_HAIRPIN) -> None:
        self.hairpin = hairpin

    def valid_pair(self, i, j):
        # Always allow None to pair with anything.
        if self.prim[i] is None or self.prim[j] is None:
            return True
        return valid_pair(self.prim[i], self.prim[j])

    def precomp(self, prim: str):
        """
        Performs an :math:`\mathcal{O}(n^3)` precomputation algorithm to permit
        efficient structure sampling. Set an element of the input sequence to `None`
        to allow any pairs to that position.

        Args:
          prim: A primary RNA sequence.
        """
        self.prim = prim
        self.dp = [[0]*len(prim) for _ in range(len(prim))]
        for i in range(len(prim)):
            self.dp[i][i] = 1
        for i in range(len(prim)-1, -1, -1):
            for j in range(i+1, len(prim)):
                self.dp[i][j] += self.dp[i+1][j]
                for k in range(i+self.hairpin+1, j+1):
                    if not self.valid_pair(i, k):
                        continue
                    self.dp[i][j] += (self.dp[i+1][k-1] if i+1 <
                                      k-1 else 1)*(self.dp[k+1][j] if k+1 < j else 1)

    def count_structures(self):
        """
        Counts the number of precomputed structures.
        """
        return self.dp[0][len(self.prim)-1]

    def get_nth(self, n: int) -> list:
        """
        Retrieves the :math:`n^{th}` structure where :math:`n \in [0, |S|]` with `|S| = self.count_structures()`.

        Args:
          n: The structure index.

        Returns:
          A secondary structure in *matching* format.
        """
        match = [i for i in range(len(self.prim))]

        def trace(i, j, n):
            if i >= j:
                return
            if n < self.dp[i+1][j]:
                trace(i+1, j, n)
                return
            n -= self.dp[i+1][j]
            for k in range(i+self.hairpin+1, j+1):
                if not self.valid_pair(i, k):
                    continue
                left = self.dp[i+1][k-1] if i+1 < k-1 else 1
                right = self.dp[k+1][j] if k+1 < j else 1
                if n < left*right:
                    match[i] = k
                    match[k] = i
                    trace(i+1, k-1, n//right)
                    trace(k+1, j, n % right)
                    return
                n -= left*right
        trace(0, len(self.prim)-1, n)
        return match
