"""
Basic unit test.
"""


from alpsqutip.model import build_spin_chain
from alpsqutip.operators import (LocalOperator,
                                 OneBodyOperator,
                                 ProductOperator,
                                 SumOperator,
                                 QutipOperator)

CHAIN_SIZE = 6

system_descriptor = build_spin_chain(CHAIN_SIZE)
sites = tuple(s for s in system_descriptor.sites.keys())

sxA = system_descriptor.site_operator(f"Sx@{sites[0]}")
syA = .5*1j*(system_descriptor.site_operator(f"Splus@{sites[0]}") -
             system_descriptor.site_operator(f"Sminus@{sites[0]}"))
szA = system_descriptor.site_operator(f"Sz@{sites[0]}")

syB = .5*1j*(system_descriptor.site_operator(f"Splus@{sites[1]}") -
             system_descriptor.site_operator(f"Sminus@{sites[1]}"))
szC = system_descriptor.site_operator(f"Sz@{sites[2]}")
sxA2 = sxA*sxA
sxAsyB = sxA*syB
sxAsyB_times_2 = 2 * sxAsyB
opglobal = szC + sxAsyB_times_2


def check_operator_equality(op1, op2):
    """check if two operators are numerically equal"""
    op_diff = op1-op2
    return (op_diff.dag()*op_diff).tr() < 1.e-9


def test_type_operator():
    """Tests for operator types"""
    assert isinstance(sxA, LocalOperator)
    assert isinstance(syB, LocalOperator)
    assert isinstance(szC, LocalOperator)
    assert isinstance(2*syB, LocalOperator)
    assert isinstance(syB*2, LocalOperator)
    assert isinstance(sxA+syB, OneBodyOperator)
    assert isinstance(sxA+syB+szC, OneBodyOperator)
    assert isinstance(sxA+syB+sxA*szC, SumOperator)
    assert isinstance(3.*(sxA+syB), OneBodyOperator)
    assert isinstance((sxA+syB)*2., OneBodyOperator)
    assert isinstance(syB+sxA, OneBodyOperator)
    assert isinstance(sxAsyB, ProductOperator)
    assert len(sxAsyB.sites_op) == 2
    assert isinstance(sxAsyB_times_2, ProductOperator)
    assert isinstance(opglobal, SumOperator)
    assert isinstance(sxA+syB, SumOperator)
    assert len(sxAsyB_times_2.sites_op) == 2

    opglobal.prefactor = 2
    assert sxA2.prefactor == 1
    assert opglobal.prefactor == 2

    assert check_operator_equality(sxA, sxA.to_qutip())
    terms = [sxA, syA, szA]
    assert check_operator_equality(
        sum(terms), sum(t.to_qutip() for t in terms))
    assert check_operator_equality(sxA.inv(), sxA.to_qutip().inv())
    opglobal_offset = opglobal + 1.3821
    assert check_operator_equality(
        opglobal_offset.inv(), opglobal_offset.to_qutip().inv())




def test_inv_operator():
    """test the exponentiation of different kind of operators"""
    sxA_inv = sxA.inv()
    assert isinstance(sxA_inv, LocalOperator)
    assert check_operator_equality(sxA_inv.to_qutip(), sxA.to_qutip().inv())

    sx_obl = sxA+syB+szC
    sx_obl_inv = sx_obl.inv()
    assert isinstance(sx_obl_inv, QutipOperator)
    assert check_operator_equality(
        sx_obl_inv.to_qutip(), sx_obl.to_qutip().inv())

    s_prod = sxA*syB*szC
    s_prod_inv = s_prod.inv()
    assert isinstance(s_prod, ProductOperator)
    assert check_operator_equality(
        s_prod_inv.to_qutip(), s_prod.to_qutip().inv())

    opglobal_offset = opglobal + 1.3821
    opglobal_offset_inv = opglobal_offset.inv()
    assert isinstance(opglobal_offset_inv, QutipOperator)
    assert check_operator_equality(
        opglobal_offset_inv .to_qutip(), opglobal_offset.to_qutip().inv())


def test_exp_operator():
    """test the exponentiation of different kind of operators"""
    sxA_exp = sxA.expm()
    assert isinstance(sxA_exp, LocalOperator)
    assert check_operator_equality(sxA_exp.to_qutip(), sxA.to_qutip().expm())

    sx_obl = sxA+syB+szC
    sx_obl_exp = sx_obl.expm()
    assert isinstance(sx_obl_exp, ProductOperator)
    assert check_operator_equality(
        sx_obl_exp.to_qutip(), sx_obl.to_qutip().expm())

    opglobal_exp = opglobal.expm()
    assert isinstance(opglobal_exp, QutipOperator)
    assert check_operator_equality(
        opglobal_exp.to_qutip(), opglobal.to_qutip().expm())


def test_local_operator():
    """Tests for local operators"""

    assert (szA*szA).tr() == .5 * 2**(CHAIN_SIZE-1)
    assert ((sxA*syA-syA*sxA)*szA).tr() == -1j * .5 * 2**(CHAIN_SIZE-1)
    assert (szA*(sxA*syA-syA*sxA)).tr() == -1j * .5 * 2**(CHAIN_SIZE-1)

    assert (szA*syB*szA*syB).tr() == .25 * 2**(CHAIN_SIZE-2)
    assert ((sxA*syA*syB-syA*sxA*syB)*(szA*syB)).tr() == - \
        1j * .25 * 2**(CHAIN_SIZE-2)
    assert ((szA*syB)*(sxA*syA*syB-syA*sxA*syB)).tr() == - \
        1j * .25 * 2**(CHAIN_SIZE-2)

    assert sxA.tr() == 0.
    assert (sxA2).tr() == .5 * 2**(CHAIN_SIZE-1)
    assert (szC * szC).tr() == .5 * 2**(CHAIN_SIZE-1)

    sxA_qt = sxA.to_qutip()
    sxA2_qt = sxA2.to_qutip()

    assert sxA_qt.tr() == 0
    assert (sxA2_qt).tr() == .5 * 2**(CHAIN_SIZE-1)
    assert sxA.partial_trace((sites[0],)).tr() == 0.
    assert sxA.partial_trace((sites[1],)).tr() == 0.
    assert sxA.partial_trace((sites[0], sites[1])).tr() == 0.
    assert sxA.partial_trace((sites[1], sites[2])).tr() == 0.


def test_product_operator():
    """Tests for product operators"""

    print("sxA")
    print(sxA)
    print("syB")
    print(syB)

    print("sxA syB")
    print(sxAsyB)
    print("Global op")
    print(opglobal)

    assert (sxAsyB*sxA*syB).tr() == .25 * 2**(CHAIN_SIZE-2)
    assert (opglobal*sxA*syB).tr() == .5 * 2**(CHAIN_SIZE-2)
    assert (sxAsyB_times_2 * sxAsyB_times_2).tr() == 2**(CHAIN_SIZE-2)
    assert (opglobal*opglobal).tr() == 2**(CHAIN_SIZE-2) * 2

    sxA_qt = sxA.to_qutip()
    syB_qt = syB.to_qutip()
    szC_qt = szC.to_qutip()

    sxAsyB_qt = sxAsyB.to_qutip()
    sxAsyB_times_2_qt = sxAsyB_times_2.to_qutip()
    opglobal_qt = opglobal.to_qutip()

    assert (sxAsyB_qt*sxA_qt*syB_qt).tr() == .25 * 2**(CHAIN_SIZE-2)
    assert (opglobal_qt*sxA_qt*syB_qt).tr() == .5 * 2**(CHAIN_SIZE-2)
    assert (szC_qt * szC_qt).tr() == .5 * 2**(CHAIN_SIZE-1)
    assert (sxAsyB_times_2_qt * sxAsyB_times_2_qt).tr() == 2**(CHAIN_SIZE-2)
    assert (opglobal_qt*opglobal_qt).tr() == 2**(CHAIN_SIZE-2) * 2


def test_qutip_operators():
    """Test for the quip representation"""

    sxA_qt = sxA.to_qutip_operator()
    sxA2_qt = sxA_qt * sxA_qt

    syB_qt = syB.to_qutip_operator()
    szC_qt = szC.to_qutip_operator()
    sxAsyB_qt = sxA_qt * syB_qt
    sxAsyB_times_2_qt = 2 * sxAsyB_qt
    opglobal_qt = szC_qt + sxAsyB_times_2_qt

    subsystems = [[sites[0]], [sites[1]], [
        sites[0], [sites[1]]], [sites[1], [sites[2]]]]

    for subsystem in subsystems:
        assert (sxA_qt).partial_trace(subsystem).tr() == 0.
        assert (sxA2_qt).partial_trace(
            subsystem).tr() == .5 * 2**(CHAIN_SIZE-1)
        assert (sxAsyB_qt*sxA_qt *
                syB_qt).partial_trace(subsystem).tr() == .25 * 2**(CHAIN_SIZE-2)
        assert (opglobal_qt*sxA_qt *
                syB_qt).partial_trace(subsystem).tr() == .5 * 2**(CHAIN_SIZE-2)
        assert (szC_qt * szC_qt).partial_trace(subsystem).tr() == .5 * \
            2**(CHAIN_SIZE-1)
        assert (sxAsyB_times_2_qt *
                sxAsyB_times_2_qt).partial_trace(subsystem).tr() == 2**(CHAIN_SIZE-2)
        assert (
            opglobal_qt*opglobal_qt).partial_trace(subsystem).tr() == 2**(CHAIN_SIZE-2) * 2
        assert (opglobal_qt*sxA_qt).partial_trace(subsystem).tr() == 0.
        assert (
            opglobal_qt*opglobal).partial_trace(subsystem).tr() == 2**(CHAIN_SIZE-2) * 2
        assert (
            opglobal*opglobal_qt).partial_trace(subsystem).tr() == 2**(CHAIN_SIZE-2) * 2

    # Tests for QutipOperators defined without a system
    detached_qutip_operator = QutipOperator(sxAsyB_times_2_qt.operator)
    assert ((sxAsyB_times_2_qt.operator)**2).tr() == 2**(CHAIN_SIZE-2)
    assert (detached_qutip_operator *
            detached_qutip_operator).tr() == 2**(CHAIN_SIZE-2)

    detached_qutip_operator = QutipOperator(sxAsyB_times_2_qt.operator, names={
                                            s: i for i, s in enumerate(sites)})
    assert (detached_qutip_operator *
            detached_qutip_operator).partial_trace(sites[0]).tr() == 2**(CHAIN_SIZE-2)
