from utils import RNA_ALPHA, matching_to_db
from sampling import UniformStructureSampler

def ss_partition(p_seq, energy_fn):
    n = p_seq.shape[0]

    def all_substruc_boltz_sum_uss(seq):
        uss = UniformStructureSampler()
        uss.precomp(seq)
        sm = 0
        for i in range(uss.count_structures()):
            sm += energy_fn(seq, uss.get_nth(i))
        return sm

    def seq_prob(seq):
        seq_prob = 1
        for i in range(len(seq)):
            seq_prob *= p_seq[i][RNA_ALPHA.index(seq[i])]
        return seq_prob

    def f(seq_list):
        if len(seq_list) == n:
            seq = ''.join(seq_list)
            return seq_prob(seq) * all_substruc_boltz_sum_uss(seq)
        sm = 0
        for b in RNA_ALPHA:
            sm += f(seq_list + [b])
        return sm

    return f([])
