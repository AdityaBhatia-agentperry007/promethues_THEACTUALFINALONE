"""
prometheus.crypto.pir
=====================
Two-server Private Information Retrieval (PIR). A client fetches record j from a
database replicated across two non-colluding servers, and NEITHER server learns j.

Two implementations, both real and single-server-private:
  - cgks_pir : information-theoretic, O(n)-bit query. Always correct. The fallback.
  - dpf_pir  : O(log n * lambda)-bit query via the Boyle-Gilboa-Ishai (2,2)-DPF.
               This is the exact primitive underlying Hafiz-Henry / PIRSONA:
               the per-party leaf "flags" satisfy flags0 XOR flags1 = e_alpha
               (Observation 1, Vadapalli-Bayatbabolghani-Henry, PoPETs 2021.4).

No third-party dependencies (stdlib hashlib/secrets only). For production one would
swap the SHAKE256 PRG for fixed-key AES, as PIRSONA does; the protocol logic is identical.
"""
from __future__ import annotations
import hashlib, secrets
from typing import List, Tuple

LAMBDA = 16  # 128-bit seeds

def _xor(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))

# ----------------------------- CGKS 2-server PIR -----------------------------
def cgks_query(n: int, j: int) -> Tuple[bytes, bytes]:
    q0 = bytes(b & 1 for b in secrets.token_bytes(n))   # uniform bit per index
    q1 = bytearray(q0); q1[j] ^= 1                        # symmetric difference = {j}
    return q0, bytes(q1)

def cgks_answer(db: List[bytes], q: bytes) -> bytes:
    w = len(db[0]); acc = bytearray(w)
    for i, bit in enumerate(q):
        if bit: acc = bytearray(_xor(acc, db[i]))
    return bytes(acc)

def cgks_pir(db: List[bytes], j: int):
    q0, q1 = cgks_query(len(db), j)
    a0, a1 = cgks_answer(db, q0), cgks_answer(db, q1)
    return _xor(a0, a1), {"server0_view": q0, "server1_view": q1}

# ----------------------------- BGI (2,2)-DPF -----------------------------
def _G(seed: bytes):
    out = hashlib.shake_256(b"DPF" + seed).digest(2 * (LAMBDA + 1))
    return out[0:LAMBDA], out[LAMBDA] & 1, out[LAMBDA+1:2*LAMBDA+1], out[2*LAMBDA+1] & 1

def _smul(bit: int, s: bytes) -> bytes:
    return s if bit else bytes(LAMBDA)

def dpf_gen(alpha: int, h: int):
    s0, s1 = secrets.token_bytes(LAMBDA), secrets.token_bytes(LAMBDA)
    root0, root1 = s0, s1
    t0, t1 = 0, 1
    CW = []
    for i in range(h):
        sL0, tL0, sR0, tR0 = _G(s0)
        sL1, tL1, sR1, tR1 = _G(s1)
        bit = (alpha >> (h - 1 - i)) & 1
        s_lose0, s_lose1 = (sR0, sR1) if bit == 0 else (sL0, sL1)
        sCW = _xor(s_lose0, s_lose1)
        tCW_L = tL0 ^ tL1 ^ bit ^ 1
        tCW_R = tR0 ^ tR1 ^ bit
        CW.append((sCW, tCW_L, tCW_R))
        if bit == 0:
            s0, t0 = _xor(sL0, _smul(t0, sCW)), tL0 ^ (t0 & tCW_L)
            s1, t1 = _xor(sL1, _smul(t1, sCW)), tL1 ^ (t1 & tCW_L)
        else:
            s0, t0 = _xor(sR0, _smul(t0, sCW)), tR0 ^ (t0 & tCW_R)
            s1, t1 = _xor(sR1, _smul(t1, sCW)), tR1 ^ (t1 & tCW_R)
    return (root0, 0, CW), (root1, 1, CW)

def dpf_eval(key, x: int, h: int) -> int:
    root, t, CW = key
    s = root
    for i in range(h):
        sL, tL, sR, tR = _G(s)
        sCW, tCW_L, tCW_R = CW[i]
        sL, tL = _xor(sL, _smul(t, sCW)), tL ^ (t & tCW_L)
        sR, tR = _xor(sR, _smul(t, sCW)), tR ^ (t & tCW_R)
        if (x >> (h - 1 - i)) & 1: s, t = sR, tR
        else:                      s, t = sL, tL
    return t

def dpf_evalfull(key, n: int, h: int) -> List[int]:
    return [dpf_eval(key, x, h) for x in range(n)]

def dpf_pir(db: List[bytes], j: int):
    n = len(db); h = max(1, (n - 1).bit_length())
    k0, k1 = dpf_gen(j, h)
    f0, f1 = dpf_evalfull(k0, n, h), dpf_evalfull(k1, n, h)
    w = len(db[0]); a0, a1 = bytearray(w), bytearray(w)
    for i in range(n):
        if f0[i]: a0 = bytearray(_xor(a0, db[i]))
        if f1[i]: a1 = bytearray(_xor(a1, db[i]))
    return _xor(bytes(a0), bytes(a1)), {"server0_seed": k0[0], "server1_seed": k1[0],
                                        "server0_flags": f0, "server1_flags": f1}

