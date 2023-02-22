def float_hash(seed, *args):
    base = 97
    mod = 10007
    p = 1
    v = (seed*7777777)%mod
    for a in args:
        if a is None:
            continue
        v += p*a
        v %= mod
        p *= base
        p %= mod
    v = (v * 9999999) % mod
    return v/(mod-1)
