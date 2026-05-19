import secrets
from backend.crypto import pir as P

rng = secrets.SystemRandom()

def _trials(fn, N=300):
    for _ in range(N):
        n = rng.randint(2, 90); w = rng.randint(1, 48)
        db = [secrets.token_bytes(w) for _ in range(n)]
        j = rng.randint(0, n - 1)
        rec, _ = fn(db, j)
        assert rec == db[j], (fn.__name__, n, j)

def test_cgks_correct():
    _trials(P.cgks_pir)

def test_dpf_correct():
    _trials(P.dpf_pir)

def test_dpf_onehot():
    for _ in range(300):
        n = rng.randint(2, 90); h = max(1, (n - 1).bit_length()); a = rng.randint(0, n - 1)
        k0, k1 = P.dpf_gen(a, h)
        oh = [P.dpf_eval(k0, x, h) ^ P.dpf_eval(k1, x, h) for x in range(n)]
        assert oh == [1 if x == a else 0 for x in range(n)]

def test_single_server_view_independent_of_index():
    # Structural: a CGKS server view is a fresh uniform bitvector sampled before j is used,
    # so its distribution is identical for any j. Sanity-check uniform balance.
    n = 64
    ones = 0; total = 0
    for _ in range(2000):
        q0, _ = P.cgks_query(n, rng.randint(0, n - 1))
        ones += sum(q0); total += n
    frac = ones / total
    assert 0.45 < frac < 0.55  # ~uniform, no dependence on j leaks into the marginal

