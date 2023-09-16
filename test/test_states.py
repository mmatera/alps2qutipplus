"""
Basic unit test for states.
"""


import qutip
from alpsqutip.model import SystemDescriptor, build_spin_chain
from alpsqutip.operators import OneBodyOperator, ProductOperator, SumOperator
from alpsqutip.states import (
    GibbsDensityOperator,
    GibbsProductDensityOperator,
    ProductDensityOperator,
)

from .helper import check_equality, expect_from_qutip


# from alpsqutip.settings import VERBOSITY_LEVEL


CHAIN_SIZE = 6

system: SystemDescriptor = build_spin_chain(CHAIN_SIZE)
sites: tuple = tuple(s for s in system.sites.keys())

sz_total: OneBodyOperator = system.global_operator("Sz")
hamiltonian: SumOperator = system.global_operator("Hamiltonian")

global_identity: ProductOperator = ProductOperator({}, 1.0, system)
sx_A = ProductOperator({sites[0]: qutip.sigmax()}, 1.0, system)
sx_B = ProductOperator({sites[1]: qutip.sigmax()}, 1.0, system)
sx_AB = 0.7 * sx_A + 0.3 * sx_B

sz_A = ProductOperator({sites[0]: qutip.sigmaz()}, 1.0, system)
sz_B = ProductOperator({sites[1]: qutip.sigmaz()}, 1.0, system)
sz_AB = 0.7 * sz_A + 0.3 * sz_B


sh_A = 0.25 * sx_A + 0.5 * sz_A
sh_B = 0.25 * sx_B + 0.5 * sz_B
sh_AB = 0.7 * sh_A + 0.3 * sh_B


subsystem_1 = [sites[0], sites[1]]
subsystem_2 = [sites[0], sites[2]]


def test_states():
    """Tests for state objects"""
    # enumerate the name of each subsystem
    assert type(sz_total) is OneBodyOperator

    test_cases_states = {}
    observable_cases = {
        "Identity": ProductOperator({}, 1.0, system),
        "sz_total": sz_total,
        "sx_A": sx_A,
        "sz_B": sz_B,
        "sh_AB": sh_AB,
        "hamiltonian": hamiltonian,
        "observable array": [[sh_AB, sh_A], [sz_A, sx_A]],
    }

    test_cases_states["fully mixed"] = ProductDensityOperator({}, 1.,
                                                              system=system)

    test_cases_states["gibbs_sz"] = GibbsProductDensityOperator(
        sz_total, system=system)

    test_cases_states["gibbs_sz_as_product"] = GibbsProductDensityOperator(
        sz_total, system=system
    ).to_product_state()
    test_cases_states["gibbs_sz_bar"] = GibbsProductDensityOperator(
        -sz_total, system=system
    )
    test_cases_states["gibbs_H"] = GibbsDensityOperator(
        hamiltonian, system=system)
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
        print("\n", 100 * "@", "\n", name, "\n", 100 * "@")
        assert abs(rho.tr() - 1) < 1.0e-10
        assert abs(1 - qt_test_cases[name].tr()) < 1.0e-10

        for subsystem in [subsystem_1, subsystem_2]:
            assert check_equality(rho.partial_trace(subsystem).tr(), 1)

        # Check Expectation Values

        expectation_values = rho.expect(observable_cases)
        qt_expectation_values = expect_from_qutip(
            qt_test_cases[name], observable_cases)

        assert isinstance(expectation_values, dict)
        assert isinstance(qt_expectation_values, dict)
        for obs in expectation_values:
            print("\n     ", 80 * "*", "\n     ", name, obs)
            print(expectation_values)
            print(qt_expectation_values)
            assert check_equality(
                expectation_values[obs], qt_expectation_values[obs])


# test_load()
# test_all()
# test_eval_expr()
