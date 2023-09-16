"""
Basic unit test for states.
"""


from alpsqutip.operators import OneBodyOperator
from alpsqutip.states import (
    GibbsDensityOperator,
    GibbsProductDensityOperator,
    ProductDensityOperator,
)

from .helper import (
    alert,
    check_equality,
    expect_from_qutip,
    hamiltonian,
    observable_cases,
    subsystems,
    system,
    sz_total,
)

# from alpsqutip.settings import VERBOSITY_LEVEL


def test_states():
    """Tests for state objects"""
    # enumerate the name of each subsystem
    assert isinstance(sz_total, OneBodyOperator)

    test_cases_states = {}

    test_cases_states["fully mixed"] = ProductDensityOperator({}, 1.0, system=system)

    test_cases_states["gibbs_sz"] = GibbsProductDensityOperator(sz_total, system=system)

    test_cases_states["gibbs_sz_as_product"] = GibbsProductDensityOperator(
        sz_total, system=system
    ).to_product_state()
    test_cases_states["gibbs_sz_bar"] = GibbsProductDensityOperator(
        -sz_total, system=system
    )
    test_cases_states["gibbs_H"] = GibbsDensityOperator(hamiltonian, system=system)
    test_cases_states["gibbs_H"] = (
        test_cases_states["gibbs_H"] / test_cases_states["gibbs_H"].tr()
    )
    test_cases_states["mixture"] = (
        0.5 * test_cases_states["gibbs_H"]
        + 0.25 * test_cases_states["gibbs_sz"]
        + 0.25 * test_cases_states["gibbs_sz_bar"]
    )

    qt_test_cases = {
        name: operator.to_qutip() for name, operator in test_cases_states.items()
    }

    for name, rho in test_cases_states.items():
        alert(0, "\n", 100 * "@", "\n", name, "\n", 100 * "@")
        assert abs(rho.tr() - 1) < 1.0e-10
        assert abs(1 - qt_test_cases[name].tr()) < 1.0e-10

        for subsystem in subsystems:
            assert check_equality(rho.partial_trace(subsystem).tr(), 1)

        # Check Expectation Values

        expectation_values = rho.expect(observable_cases)
        qt_expectation_values = expect_from_qutip(qt_test_cases[name], observable_cases)

        assert isinstance(expectation_values, dict)
        assert isinstance(qt_expectation_values, dict)
        for obs in expectation_values:
            alert(0, "\n     ", 80 * "*", "\n     ", name, obs)
            alert(0, expectation_values)
            alert(0, qt_expectation_values)
            assert check_equality(expectation_values[obs], qt_expectation_values[obs])


# test_load()
# test_all()
# test_eval_expr()
