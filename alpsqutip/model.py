"""
Define SystemDescriptors and different kind of operators
"""

from functools import reduce
from typing import Optional
import qutip
from qutip import Qobj
from alpsqutip.geometry import GraphDescriptor
from alpsqutip.utils import eval_expr


class SystemDescriptor:
    """
    System Descriptor class.
    """

    def __init__(
        self,
        graph: GraphDescriptor,
        basis: dict,
        parms: Optional[dict] = None,
        sites=None,
    ):
        self.graph = graph
        self.basis = basis
        self.parms = basis.parms
        if parms is not None:
            self.parms.update(parms)
        site_basis = basis.site_basis
        if sites:
            self.sites = sites
        else:
            self.sites = {
                node: site_basis[attr["type"]]
                for node, attr in graph.nodes.items()
            }

        self.dimensions = {
            name: site["dimension"] for name, site in self.sites.items()
        }
        self.site_operators = {}
        self.bond_operators = {}
        self.global_operators = {}
        self._load_site_operators()
        self._load_global_ops()

    def __repr__(self):
        result = (
            "graph:"
            + repr(self.graph)
            + "\n"
            + "sites:"
            + repr(self.sites.keys())
            + "\n"
            + "dimensions:"
            + repr(self.dimensions)
        )
        return result

    def subsystem(self, sites: list):
        """
        Build a subsystem including the sites listed
        in sites
        """
        parms = self.parms.copy()
        basis = self.basis
        graph = self.graph.subgraph(sites)
        return SystemDescriptor(graph, basis, parms)

    def _load_site_operators(self):
        for site_name, site in self.sites.items():
            for op_name in site["operators"]:
                op_site = f"{op_name}@{site_name}"
                self.site_operator(op_site)

    def _load_global_ops(self):
        # First, load conserved quantum numbers:

        for constraint_qn in self.basis.constraints:
            global_qn = ProductOperator({}, 0.0, self)
            for site, site_basis in self.sites.items():
                local_qn = site_basis["qn"].get(constraint_qn, None)
                if local_qn is None:
                    continue
                op_name = local_qn["operator"]
                operator = site_basis["operators"][op_name]
                global_qn = global_qn + \
                    ProductOperator({site: operator}, 1, self)

            if not isinstance(global_qn, ProductOperator):
                self.global_operators[constraint_qn] = global_qn

        names = list(self.basis.global_ops)
        for gop in names:
            self.global_operator(gop)

    def site_operator(self, name: str, site: str = "") -> "ProductOperator":
        """
        Return a global operator representing an operator `name`
        acting over the site `site`. By default, the name is assumed
        to specify both the name and site in the form `"name@site"`.
        """
        if site != "":
            op_name = name
            name = f"{name}@{site}"
        else:
            op_name, site = name.split("@")

        site_op = self.site_operators.get(site, {}).get(name, None)
        if site_op is not None:
            return site_op

        local_op = self.sites[site]["operators"].get(op_name, None)
        if local_op is None:
            return None
        result_op = ProductOperator({site: local_op}, 1.0, self)
        self.site_operators.setdefault(site, {})
        self.site_operators[site][op_name] = result_op
        return result_op

    def bond_operator(
        self, name: str, src: str, dst: str, skip=None
    ) -> "NBodyOperator":
        """Bond operator by name and sites"""

        result_op = self.bond_operators.get(
            (
                name,
                src,
                dst,
            ),
            None,
        )
        if result_op is not None:
            return result_op
        # Try to build the bond operator from the descriptors.
        bond_op_descriptors = self.basis.bond_ops
        bond_op_descriptor = bond_op_descriptors.get(name, None)
        if bond_op_descriptor is None:
            return None

        bond_dependencies = [
            bop
            for bop in bond_op_descriptors
            if bond_op_descriptor.find(bop + "@") >= 0
        ]
        bond_op_descriptor = bond_op_descriptor.replace("@", "__")
        src_operators = self.sites[src]["operators"]
        dst_operators = self.sites[dst]["operators"]

        # Load site operators on src and dst
        parms_and_ops = {
            f"{name_src}__src": self.site_operator(name_src, src)
            for name_src in src_operators
        }
        parms_and_ops.update(
            {
                f"{name_dst}__dst": self.site_operator(name_dst, dst)
                for name_dst in dst_operators
            }
        )
        parms_and_ops.update(self.parms)

        # Try to evaluate
        result = eval_expr(bond_op_descriptor, parms_and_ops)
        if not (result is None or isinstance(result, str)):
            self.bond_operators[
                (
                    name,
                    src,
                    dst,
                )
            ] = result
            return result

        # Now, try to include existent bond operators
        parms_and_ops.update(
            {
                f"{tup_op[0]}__src_dst": op
                for tup_op, op in self.bond_operators.items()
                if tup_op[0] == src and tup_op[1] == dst
            }
        )
        parms_and_ops.update(
            {
                f"{tup_op[0]}__dst_src": op
                for tup_op, op in self.bond_operators.items()
                if tup_op[0] == dst and tup_op[1] == src
            }
        )

        result = eval_expr(bond_op_descriptor, parms_and_ops)
        if not (result is None or isinstance(result, str)):
            self.bond_operators[tuple(name, src, dst)] = result
            return result

        # Finally, try to load other operators
        if skip is None:
            skip = [name]
        else:
            skip.append(name)
        for bop in bond_dependencies:
            # Skip this name
            if bop in skip:
                continue

            # src->dst
            new_bond_op = self.bond_operator(bop, src, dst, skip)
            if new_bond_op is None:
                continue

            parms_and_ops[f"{bop}__src_dst"] = new_bond_op

            new_bond_op = self.bond_operator(bop, dst, src, skip)
            if new_bond_op is None:
                continue

            parms_and_ops[f"{bop}__dst_src"] = new_bond_op

            result = eval_expr(bond_op_descriptor, parms_and_ops)
            if result is not None and not isinstance(result, str):
                self.bond_operators[
                    (
                        name,
                        src,
                        dst,
                    )
                ] = result
                return result

        # If this fails after exploring all the operators, then it means that
        # the operator is not in the basis.
        # if skip[-1]==name:
        #    self.bond_operators[(name, src, dst,)] = None
        return None

    def global_operator(self, name):
        """Return a global operator by its name"""
        result = self.global_operators.get(name, None)
        if result is not None:
            return result
        # Build the global_operator from the descriptor
        op_descr = self.basis.global_ops.get(name, None)
        if op_descr is None:
            return None

        site_terms_descr = op_descr["site terms"]
        bond_terms_descr = op_descr["bond terms"]
        site_terms = []
        bond_terms = []
        # Process site terms

        for term in site_terms_descr:
            expr = term["expr"]
            site_type = term.get("type", None)
            t_parm = {}
            t_parm.update(term.get("parms", {}))
            t_parm.update(self.parms)
            for node_name, node in self.graph.nodes.items():
                node_type = node.get("type", None)
                if site_type is not None and site_type != node_type:
                    continue
                s_expr = expr.replace("#", node_type).replace("@", "__")
                s_parm = {
                    key.replace("#", node_type): val
                    for key, val in t_parm.items()
                }
                s_parm.update(
                    {
                        f"{name_op}_local": local_op
                        for name_op, local_op in self.site_operators[
                            node_name
                        ].items()
                    }
                )
                term_op = eval_expr(s_expr, s_parm)
                if term_op is None or isinstance(term_op, str):
                    print("<<", s_expr, ">>",
                          "could not be evaluated. Aborting evaluation of ", name)
                    self.basis.global_ops.pop(name)
                    return None
                site_terms.append(term_op)

        # Process bond terms
        for term in bond_terms_descr:
            expr = term["expr"]
            term_type = term.get("type", None)
            t_parm = {}
            t_parm.update(term.get("parms", {}))
            t_parm.update(self.parms)
            for edge_type, edges in self.graph.edges.items():
                if term_type is not None and term_type != edge_type:
                    continue
                e_expr = expr.replace("#", edge_type).replace("@", "__")
                for src, dst in edges:
                    e_parms = {
                        key.replace("#", f"{edge_type}"): val
                        for key, val in t_parm.items()
                    }
                    e_parms.update(
                        {
                            f"{name}__src": val
                            for name, val in self.site_operators[src].items()
                        }
                    )
                    e_parms.update(
                        {
                            f"{name}__dst": val
                            for name, val in self.site_operators[dst].items()
                        }
                    )
                    # Try to compute using only site terms
                    term_op = eval_expr(e_expr, e_parms)
                    if not isinstance(term_op, str):
                        bond_terms.append(term_op)
                        continue
                    # Try now adding the bond operators
                    for name_bop in self.basis.bond_ops:
                        self.bond_operator(name_bop, src, dst)
                        self.bond_operator(name_bop, dst, src)

                    e_parms.update(
                        {
                            f"{key[0]}__src_dst": val
                            for key, val in self.bond_operators.items()
                            if key[1] == src and key[2] == dst
                        }
                    )
                    e_parms.update(
                        {
                            f"{key[0]}__dst_src": val
                            for key, val in self.bond_operators.items()
                            if key[2] == src and key[1] == dst
                        }
                    )
                    term_op = eval_expr(e_expr, e_parms)
                    if not isinstance(term_op, str):
                        bond_terms.append(term_op)
                    else:
                        self.basis.global_ops.pop(name)
                        return None

        result = sum(site_terms) + sum(bond_terms)
        self.global_operators[name] = result
        return result


class Operator:
    """Base class for operators"""

    def __neg__(self):
        raise NotImplementedError()

    def __sub__(self, operand):
        neg_op = -operand
        return self + neg_op

    def __radd__(self, operand):
        return self + operand

    def __rsub__(self, operand):
        neg_self = -self
        return operand + neg_self

    def _repr_latex_(self):
        """LaTeX Representation"""
        qutip_repr = self.to_qutip()
        if isinstance(qutip_repr, qutip.Qobj):
            parts = qutip_repr._repr_latex_().split("$")
            tex = parts[1] if len(parts) > 2 else "-?-"
        else:
            tex = str(qutip_repr)
        return f"${tex}$"

    def tr(self):
        """The trace of the operator"""
        return self.partial_trace([]).prefactor

    def partial_trace(self, sites: list):
        """Partial trace over sites not listed in `sites`"""
        raise NotImplementedError

    def to_qutip(self):
        """Convert to a Qutip object"""
        raise NotImplementedError


class LocalOperator(Operator):
    """
    Operator acting over a single site.
    """

    def __init__(
            self,
            site,
            local_operator,
            system: Optional[SystemDescriptor] = None,
    ):
        self.site = site
        self.operator = local_operator
        self.system = system

    def __add__(self, operand):
        site = self.site
        if isinstance(operand, LocalOperator):
            if site == operand.site:
                system = self.system or operand.system
                return LocalOperator(site, self.operator+operand.operator, system)
        return operand + self

    def __bool__(self):
        operator = self.operator
        if isinstance(operator, Qobj):
            return operator.data.count_nonzero() > 0
        return bool(self.operator)

    def __mul__(self, operand):
        if isinstance(operand, (int, float, complex)):
            return LocalOperator(self.site, operand*self.operator, self.system)
        if isinstance(operand, LocalOperator):
            site = self.site
            system = self.system or operand.system
            if site == operand.site:
                return LocalOperator(site, self.operator*operand.operator, system)
        return ProductOperator({site: self.operator}, system) * operand

    def __neg__(self):
        return LocalOperator(self.site, -self.operator, self.system)

    def __pow__(self, exp):
        return LocalOperator(self.site,
                             self.operator**exp,
                             self.system
                             )

    def __repr__(self):
        return f"Local Operator on site {self.site}:\n {repr(self.operator)}"

    def dag(self):
        """
        Return the adjoint operator
        """
        return LocalOperator(self.site, self.operator.dag(), self.system)

    def partial_trace(self, sites):
        system = self.system
        if system is None:
            if self.site in sites:
                return self
            return ProductOperator({}, self.operator.tr())

        dimensions = system.dimensions
        subsystem = system.subsystem(sites)
        local_sites = subsystem.sites
        prefactors = [d for s, d in dimensions.items() if s not in local_sites]
        prefactor = reduce(lambda x, y: x*y, prefactors)

        if self.site in local_sites:
            return LocalOperator(self.site, self.operator*prefactor, subsystem)

        return ProductOperator({}, self.operator.tr()*prefactor, subsystem)

    def to_qutip(self):
        """Convert to a Qutip object"""
        site = self.site
        dimensions = self.system.dimensions
        qutip.tensor([self.operator if s == site else qutip.qeye(d)
                     for s, d in dimensions.items()])


class ProductOperator(Operator):
    """Product of operators acting over different sites"""

    def __init__(
        self,
        sites_op: dict,
        prefactor=1.0,
        system: Optional[SystemDescriptor] = None,
    ):
        self.sites_op = sites_op
        if any(op.data.count_nonzero() == 0 for op in sites_op.values()):
            prefactor = 0
            self.sites_op = {}
        self.prefactor = prefactor
        self.system = system
        if system is not None:
            self.size = len(system.sites)
            self.dimensions = {
                name: site["dimension"] for name, site in system.sites.items()
            }

    def __add__(self, operand):
        if self.prefactor == 0:
            return operand
        if isinstance(operand, (int, float, complex)):
            if operand == 0:
                return self
            operand = ProductOperator({}, operand, self.system)
        elif isinstance(operand, ProductOperator):
            if operand.prefactor == 0:
                return self
            if len(operand.sites_op) == 1 and len(self.sites_op) == 1:
                site = next(iter(self.sites_op))
                if site in operand.sites_op:
                    return ProductOperator(
                        {
                            site: (
                                self.sites_op[site] * self.prefactor
                                + operand.sites_op[site] * operand.prefactor
                            )
                        },
                        prefactor=1,
                        system=self.system,
                    )
            new_terms = [operand, self]
        elif isinstance(operand, NBodyOperator):
            new_terms = operand.terms + [self]
        return NBodyOperator(new_terms)

    def __bool__(self):
        return bool(self.prefactor)

    def __mul__(self, operand):
        if isinstance(operand, (int, float, complex)):
            new_prefactor = self.prefactor * operand
            if new_prefactor == 0.0:
                return ProductOperator(
                    {}, prefactor=new_prefactor, system=self.system
                )
            return ProductOperator(
                self.sites_op, prefactor=new_prefactor, system=self.system
            )
        if isinstance(operand, ProductOperator):
            new_sites_op = self.sites_op.copy()
            for pos, factor in operand.sites_op.items():
                if pos in new_sites_op:
                    new_sites_op[pos] = new_sites_op[pos] * factor
                else:
                    new_sites_op[pos] = factor
            return ProductOperator(
                new_sites_op,
                prefactor=self.prefactor * operand.prefactor,
                system=self.system,
            )
        if isinstance(operand, NBodyOperator):
            new_terms = [self * op_2 for op_2 in operand.terms]
            new_terms = [term for term in new_terms if term]
            return NBodyOperator(new_terms)

        raise NotImplementedError

    def __neg__(self):
        return ProductOperator(self.sites_op, -self.prefactor, self.system)

    def __pow__(self, exp):
        return ProductOperator(
            {s: op**exp for s, op in self.sites_op.items()},
            self.prefactor**exp,
            self.system,
        )

    def __repr__(self):
        result = str(self.prefactor) + " * (\n"
        result += "\n".join(str(item) for item in self.sites_op.items())
        result += ")"
        return result

    def __rmul__(self, operand):
        if isinstance(operand, (int, float, complex)):
            return self * operand
        return NotImplementedError

    def dag(self):
        """
        Return the adjoint operator
        """
        sites_op_dag = {key: op.dag() for key, op in self.sites_op.items()}
        prefactor = self.prefactor
        if isinstance(prefactor, complex):
            prefactor = prefactor.conj()
        return ProductOperator(sites_op_dag, prefactor, self.system)

    def partial_trace(self, sites: list):
        full_system_sites = self.system.sites
        dimensions = self.dimensions
        sites_in = [s for s in sites if s in full_system_sites]
        sites_out = [s for s in full_system_sites if s not in sites_in]
        subsystem = self.system.subsystem(sites_in)
        sites_op = self.sites_op
        prefactors = [
            sites_op[s].tr() if s in sites_op else dimensions[s]
            for s in sites_out
        ]
        sites_op = {s: o for s, o in sites_op.items() if s in sites_in}
        prefactor = self.prefactor
        for factor in prefactors:
            if factor == 0:
                return ProductOperator({}, factor, subsystem)
            prefactor *= factor
        return ProductOperator(sites_op, prefactor, subsystem)

    def to_qutip(self):
        if self.prefactor == 0 or len(self.system.dimensions) == 0:
            return self.prefactor
        ops = self.sites_op
        return self.prefactor * qutip.tensor(
            [
                ops.get(site, None) if site in ops else qutip.qeye(dim)
                for site, dim in self.system.dimensions.items()
            ]
        )

    def tr(self):
        result = self.partial_trace([])
        return result.prefactor


class NBodyOperator(Operator):
    """
    Represents a linear combination of product operators
    """

    def __init__(self, terms_coeffs: list):
        self.terms = terms_coeffs
        if terms_coeffs:
            system = terms_coeffs[0].system
            self.system = system
            for term in terms_coeffs:
                term_system = term.system
                if term.system is term_system:
                    continue
                if len(term_system.dimensions) > len(system.dimensions):
                    system, term_system = term_system, system

    def __add__(self, operand):
        if isinstance(operand, (int, float, complex)):
            if operand == 0.0:
                return self
            operand = ProductOperator({}, operand, self.system)

        elif isinstance(operand, ProductOperator):
            if operand.prefactor == 0:
                return self
            new_terms = self.terms + [operand]
        elif isinstance(operand, NBodyOperator):
            if len(operand.terms) == len(self.terms) == 1:
                return self.terms[0] + operand.terms[0]
            new_terms = self.terms + operand.terms
        # TODO: cancel terms
        new_terms = [t for t in new_terms if t.prefactor != 0]
        return NBodyOperator(new_terms)

    def __bool__(self):
        if len(self.terms) == 0:
            return False

        if any(t for t in self.terms):
            return True
        return False

    def __pow__(self, exp):
        if isinstance(exp, int):
            if exp == 0:
                return 1
            if exp == 1:
                return self
            if exp > 1:
                exp -= 1
                return self * (self**exp)
            raise TypeError("NBodyOperator does not support negative powers")
        raise TypeError(
            (
                f"unsupported operand type(s) for ** or pow(): "
                f"'NBodyOperator' and '{type(exp).__name__}'"
            )
        )

    def __mul__(self, operand):
        if isinstance(operand, (int, float, complex)):
            if operand == 0:
                return operand
            new_terms = [
                operand * operand1 for operand1 in self.terms if operand1]
        elif isinstance(operand, ProductOperator):
            if operand.prefactor:
                new_terms = [operand1 *
                             operand for operand1 in self.terms if operand1]
            else:
                new_terms = []
        elif isinstance(operand, NBodyOperator):
            new_terms = [
                op_1 * op_2 for op_1 in self.terms for op_2 in operand.terms]
        else:
            raise TypeError(type(operand))

        new_terms = [t for t in new_terms if t]
        if len(new_terms) == 0:
            return 0.0
        if len(new_terms) == 1:
            return new_terms[0]
        return NBodyOperator(new_terms)

    def __neg__(self):
        return NBodyOperator([-t for t in self.terms])

    def __repr__(self):
        return "(\n" + "\n+".join(repr(t) for t in self.terms) + "\n)"

    def __rmul__(self, operand):
        if isinstance(operand, (int, float, complex)):
            return self * operand
        return NotImplementedError

    def dag(self):
        """return the adjoint operator"""
        return NBodyOperator([t.dag() for t in self.terms])

    def partial_trace(self, sites: list):
        return sum(t.partial_trace(sites) for t in self.terms)

    def to_qutip(self):
        """Produce a qutip compatible object"""
        if len(self.terms) == 0:
            return ProductOperator({}, 0, self.system).to_qutip()
        return sum(t.to_qutip() for t in self.terms)

    def tr(self):
        return sum(t.tr() for t in self.terms)






def split_nbody_terms(operator:NBodyOperator):
    """Returns a dictionary with the terms collected by the 
    number of sites over which the term acts"""
    result_dict = {}
    for term in operator.terms:
        if isinstance(term, NBodyOperator):
            for n, t in split_nbody_terms(term).items():
                result_dict[n] = result_dict.get(n,0) + t
            continue
        if isinstance(term, LocalOperator):
            result_dict[1] = result_dict.get(1, 0) + term
        if isinstance(term, ProductOperator):
            n = len(term.sites_op)
            result_dict[n] = result_dict.get(n, 0) + term
            
    return result_dict
            
            
                
def simplify_quadratic_form(operator:NBodyOperator):
    local_ops = []
    coeffs = []
    system = operator.system
    for term in operator.terms:
        s1, s2 = tuple(LocalOperator(s, o, system) for s, o in term.sites_op.items())
        
        local_ops.append(s1+s2)
        coeffs.append(term.prefactor/8)
        local_ops.append(s1-s2)
        coeffs.append(-term.prefactor/8)
        
    def scalar_product(o_1, o_2):
        return (o_1.dag()*o_2).tr()
    
    import numpy as np
    from numpy.linalg import svd
    gram = np.array([[ scalar_product(o_1, o_2)  for o_1 in localops] 
                     for o_2 in localops])
    
    u, s, vh = svd(gram, hermitian=True, full_matrices=False)
    reduced_basis = [ sum(c*old_op  for c, old_op in zip(row,local_ops) ) for row in U.transpose()]
    
        
        
                
                
    
        
    
    
    