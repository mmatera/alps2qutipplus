"""
Helper functions for pytests
"""

from numbers import Number
from typing import Iterable

import numpy as np
import qutip
from alpsqutip.model import Operator


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
