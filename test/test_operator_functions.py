"""
Basic unit test for operator functions.
"""

import numpy as np
from alpsqutip.operators import OneBodyOperator
from alpsqutip.operator_functions import (
    eigenvalues,
    spectral_norm,
    log_op,
    relative_entropy,
)

from .helper import (
    CHAIN_SIZE,
    alert,
    check_equality,
    expect_from_qutip,
    hamiltonian,
    observable_cases,
    subsystems,
    system,
    sz_total,
    test_cases_states,
)

# from alpsqutip.settings import VERBOSITY_LEVEL


def test_eigenvalues():
    """Tests eigenvalues of different operator objects"""

    spectrum = sorted(eigenvalues(sz_total))
    for s in range(6):
        assert any(abs(e_val - s+3.) < 1e-6 for e_val in spectrum)

    # Fully mixed operator
    spectrum = sorted(eigenvalues(test_cases_states["fully mixed"]))
    assert all(abs(s-.5**CHAIN_SIZE) < 1e-6 for s in spectrum)

    #  e^(sz)/Tr e^(sz)
    spectrum = sorted(eigenvalues(test_cases_states["gibbs_sz"]))
    expected_local_spectrum = np.array(np.exp(-1), np.exp(1),)
    expected_spectrum = np.array([s1*s2*s2*s4*s5*s6
                                 for s1 in expected_local_spectrum
                                 for s2 in expected_local_spectrum
                                 for s3 in expected_local_spectrum
                                 for s4 in expected_local_spectrum
                                 for s5 in expected_local_spectrum
                                 for s6 in expected_local_spectrum
                                  ])
    assert np.linalg.norm(expected_spectrum-spectrum) < 1.e-6

    e0 = min(eigenvalues(hamiltonian, sparse=True, sort="low", eigvals=10))
    print("e0=", e0)
    assert e0 == 0


# test_load()
# test_all()
# test_eval_expr()
