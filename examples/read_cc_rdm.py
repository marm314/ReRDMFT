#!/usr/bin/env python3
"""Reader for the molecular-basis CC_RDM files <name>_MO.dm1 / <name>_MO.dm2 written by ReRDMFT (UCCSD TRUE, CC_RDM TRUE).
(The Gaussian-primitive files <name>_GAU.* use the Fortran-record formats of RHO2_OPS instead.)

dm1 record: int32 p, int32 q, float64 re, float64 im           -> gamma[p,q] = <a+_p a_q>   (Hermitian part)
dm2 record: int32 p, q, r, s, float64 re, float64 im  (p<q, r<s) -> Gamma[p,q,r,s] = <a+_p a+_q a_s a_r>
Indices are 1-based, in Kramers-pair order (1,2) = (spinor, time-reversed partner), (3,4), ...; the file ends with an all-zero
record. Energy:  E_elec = sum_pq h_pq gamma_pq + 1/2 sum_pqrs <pq|rs> Gamma_pqrs  (all pqrs; Gamma antisymmetric in pq and rs).
"""
import sys
import numpy as np

def read_dm1(path):
    rec = np.dtype([('idx', '<i4', 2), ('val', '<f8', 2)])
    a = np.fromfile(path, dtype=rec)
    end = np.where((a['idx'][:, 0] == 0) & (a['idx'][:, 1] == 0))[0][0]     # terminating all-zero record
    a = a[:end]
    n = int(a['idx'].max())
    g = np.zeros((n, n), complex)
    g[a['idx'][:, 0] - 1, a['idx'][:, 1] - 1] = a['val'][:, 0] + 1j * a['val'][:, 1]
    return g

def read_dm2(path, n=None):
    rec = np.dtype([('idx', '<i4', 4), ('val', '<f8', 2)])
    a = np.fromfile(path, dtype=rec)
    end = np.where(np.all(a['idx'] == 0, axis=1))[0][0]
    a = a[:end]
    if n is None: n = int(a['idx'].max())
    G = np.zeros((n, n, n, n), complex)               # dense expansion: only sensible for small n
    p, q, r, s = (a['idx'][:, k] - 1 for k in range(4))
    v = a['val'][:, 0] + 1j * a['val'][:, 1]
    G[p, q, r, s] = v; G[q, p, r, s] = -v; G[p, q, s, r] = -v; G[q, p, s, r] = v
    return G

if __name__ == '__main__':
    base = sys.argv[1]
    g = read_dm1(base + '.dm1')    # base = '<name>_MO'
    print('n spinors', g.shape[0], ' trace gamma =', g.trace().real, ' max|gamma-gamma^H| =', abs(g - g.conj().T).max())
    print('natural occupation numbers:', np.round(np.sort(np.linalg.eigvalsh((g + g.conj().T) / 2))[::-1], 6))
    G = read_dm2(base + '.dm2', g.shape[0])
    print('2-RDM elements (dense):', G.shape, ' max|Gamma_pqrs - conj Gamma_rspq| =', abs(G - G.transpose(2, 3, 0, 1).conj()).max())
    print('sum_p Gamma_pqps vs (N-1) gamma_qs: max diff =', abs(np.einsum('pqps->qs', G) - (g.trace().real - 1) * g).max())
