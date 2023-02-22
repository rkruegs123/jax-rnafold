from jax_rnafold.common.utils import valid_pair, HAIRPIN

# Runs an O(n^3) precomp algorithm after which all structures are given an arbitrary unique ID.
# The structure for an ID can be retrieved in O(n^2) time.
# Useful for generating a uniform random sample from the space of structures for a sequence.
# Give a [None] sequence to allow any pairs.


class UniformStructureSampler:
    def __init__(self) -> None:
        pass

    def valid_pair(self, i, j):
        # Always allow None to pair with anything.
        if self.prim[i] is None or self.prim[j] is None:
            return True
        return valid_pair(self.prim[i], self.prim[j])

    def precomp(self, prim):
        self.prim = prim
        self.dp = [[0]*len(prim) for _ in range(len(prim))]
        for i in range(len(prim)):
            self.dp[i][i] = 1
        for i in range(len(prim)-1, -1, -1):
            for j in range(i+1, len(prim)):
                self.dp[i][j] += self.dp[i+1][j]
                for k in range(i+HAIRPIN+1, j+1):
                    if not self.valid_pair(i, k):
                        continue
                    self.dp[i][j] += (self.dp[i+1][k-1] if i+1 <
                                      k-1 else 1)*(self.dp[k+1][j] if k+1 < j else 1)

    def count_structures(self):
        return self.dp[0][len(self.prim)-1]

    # Gets the nth structure where n is in [0, self.count_structures()]
    def get_nth(self, n):
        match = [i for i in range(len(self.prim))]

        def trace(i, j, n):
            if i >= j:
                return
            if n < self.dp[i+1][j]:
                trace(i+1, j, n)
                return
            n -= self.dp[i+1][j]
            for k in range(i+HAIRPIN+1, j+1):
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
