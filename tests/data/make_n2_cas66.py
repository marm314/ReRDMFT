# Generates n2_cas66_1p1.txt: PySCF CASCI(6,6) reference for tests/test_casci.cpp (N2, 1.1 A, cc-pVDZ cartesian,
# RHF orbitals, no orbital optimization). Run: PYTHONPATH=<venv site-packages> python3.12 make_n2_cas66.py
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pvdz', cart=True, unit='Angstrom', verbose=0)
mf = scf.RHF(mol).run()
mc = mcscf.CASCI(mf, 6, 6)
mc.fcisolver.conv_tol = 1e-13
e, _, ci = mc.kernel()[:3]
h1, ecore = mc.get_h1eff()
h2 = ao2mo.restore(1, mc.get_h2eff(), 6)
d1, d2 = mc.fcisolver.make_rdm12(ci, 6, mc.nelecas)
with open('n2_cas66_1p1.txt', 'w') as f:
    f.write('6 6 %.15e %.15e\n' % (ecore, e))
    for a in (h1, h2, d1, d2):
        f.write(' '.join('%.15e' % x for x in np.ravel(a)) + '\n')
print('CASCI energy', repr(e), 'ecore', ecore)
