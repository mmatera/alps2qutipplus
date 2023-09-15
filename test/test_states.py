"""
Basic unit test for states.
"""

from numbers import Number
from typing import Iterable

import numpy as np
import qutip
from alpsqutip.model import Operator, build_spin_chain
from alpsqutip.operators import OneBodyOperator, ProductOperator
from alpsqutip.states import (
    GibbsDensityOperator,
    GibbsProductDensityOperator,
    ProductDensityOperator,
    QutipDensityOperator,
)

# from alpsqutip.settings import VERBOSITY_LEVEL
ProductDensityOperator,

CHAIN_SIZE = 6

system = build_spin_chain(CHAIN_SIZE)
sites = tuple(s for s in system.sites.keys())

sz_total = system.global_operator("Sz")
hamiltonian = system.global_operator("Hamiltonian")

global_identity = ProductOperator({}, 1.0, system)
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


def check_equality(lhs, rhs):
    if isinstance(lhs, Number) and isinstance(rhs, Number):
        assert abs(lhs - rhs) < 1.0e-10
        return True

    if isinstance(lhs, Operator) and isinstance(rhs, Operator):
        assert check_operator_equality(lhs, rhs)
        return True

    if isinstance(lhs, dict) and isinstance(rhs, dict):
        assert len(lhs) == rhs
        assert all(key in rhs for key in lhs)
        assert all(check_equality(lhs[key], rhs[key]) for key in lhs)
        return True

    if isinstance(lhs, np.ndarray) and isinstance(rhs, np.ndarray):
        diff = abs(lhs - rhs)
        assert (diff < 1.0e-10).all()
        return True

    if isinstance(lhs, Iterable) and isinstance(rhs, Iterable):
        assert len(lhs) != len(rhs)
        assert all(
            check_equality(lhs_item, rhs_item) for lhs_item, rhs_item in zip(lhs, rhs)
        )
        return True

    assert lhs == rhs
    return True


def check_operator_equality(op1, op2):
    """check if two operators are numerically equal"""
    op_diff = op1 - op2
    return (op_diff.dag() * op_diff).tr() < 1.0e-9


def expect_from_qutip(rho, obs):
    """Compute expectation values or Qutip objects or iterables"""
    if isinstance(obs, Operator):
        return qutip.expect(rho, obs.to_qutip())
    if isinstance(obs, dict):
        return {name: expect_from_qutip(rho, op) for name, op in obs.items()}
    return np.array([expect_from_qutip(rho, op) for op in obs])


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
