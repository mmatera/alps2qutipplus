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


def compare_spectrum(spectrum1, spectrum2):
    assert max(abs(np.array(sorted(spectrum1)) -
               np.array(sorted(spectrum2)))) < 1.e-12


def test_eigenvalues():
    """Tests eigenvalues of different operator objects"""

    spectrum = sorted(eigenvalues(sz_total))
    for s in range(6):
        assert any(abs(e_val - s+3.) < 1e-6 for e_val in spectrum)

    # Fully mixed operator
    spectrum = sorted(eigenvalues(test_cases_states["fully mixed"]))
    assert all(abs(s-.5**CHAIN_SIZE) < 1e-6 for s in spectrum)

    e0 = min(eigenvalues(hamiltonian, sparse=True, sort="low", eigvals=10))
    print("e0=", e0)
    assert abs(e0 + 3.00199535) < 1.e-6

    #  e^(sz)/Tr e^(sz)
    spectrum = sorted(eigenvalues(test_cases_states["gibbs_sz"]))
    expected_local_spectrum = np.array([np.exp(-.5), np.exp(.5)])
    expected_local_spectrum = (expected_local_spectrum /
                               sum(expected_local_spectrum))

    expected_spectrum = expected_local_spectrum.copy()
    for i in range(5):
        expected_spectrum = np.append(
            expected_spectrum * expected_local_spectrum[0],
            expected_spectrum * expected_local_spectrum[1])

    compare_spectrum(expected_spectrum, spectrum)


# test_load()
# test_all()
# test_eval_expr()
