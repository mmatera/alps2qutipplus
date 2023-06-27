from types import FunctionType, ModuleType

import numpy as np
from numpy.random import rand


def eval_expr(expr: str, parms: dict):
    """
    Evaluate the expression `expr` replacing the variables defined in `parms`.
    expr can include python`s arithmetic expressions, and some elementary
    functions.
    """
    # TODO: Improve the workflow in a way that numpy functions
    # and constants be loaded just if they are needed.
    default_parms = {
        "pi": 3.1415926,
        "e": 2.71828183,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "rand": rand,
    }

    default_parms.update(parms)
    parms = default_parms

    if isinstance(expr, str):
        try:
            return float(expr)
        except (ValueError, TypeError):
            try:
                return complex(expr)
            except (ValueError, TypeError):
                pass

        value = parms.pop(expr, None)
        if value is not None:
            if isinstance(value, str):
                return eval_expr(value, parms)
            return value

    p_vars = [k for k in parms]
    for k in p_vars:
        if k in default_parms or isinstance(
            parms[k], (FunctionType, ModuleType, np.ufunc)
        ):
            continue
        val = parms.pop(k)
        try:
            nval = eval_expr(val, parms)
        except RecursionError:
            raise
        if not check_numeric(nval):
            raise ValueError(val)
        parms[k] = nval

    if isinstance(expr, str):
        try:
            return eval(expr, parms)
        except NameError:
            pass
    return expr


def find_ref(node, root):
    node_items = dict(node.items())
    if "ref" in node_items:
        name_ref = node_items["ref"]
        for refnode in root.findall("./" + node.tag):
            if ("name", name_ref) in refnode.items():
                return refnode
    return node


def next_name(dictionary: dict, s: int = 1, prefix: str = "") -> str:
    """
    Produces a new key for the `dictionary` with a
    `prefix`
    """
    name = f"{prefix}{s}"
    if name in dictionary:
        return next_name(dictionary, s + 1, prefix)
    return name
