import qutip

from alpsqutip.geometry import GraphDescriptor
from alpsqutip.utils import eval_expr

class SystemDescriptor:
    def __init__(self, graph: GraphDescriptor, basis: dict, parms: dict):
        self.graph = graph
        self.basis = basis
        self.parms = basis.parms
        self.parms.update(parms)
        site_basis = basis.site_basis
        self.sites = {
            node: site_basis[attr["type"]] for node, attr in graph.nodes.items()
        }
   
        self.dimensions = {name:site["dimension"]  for name, site in self.sites.items()}
        self.site_operators = {}
        self.bond_operators = {}
        self.global_operators = {}
        self._load_site_operators()
        self._load_global_ops()

    def _load_site_operators(self):
        for site_name, site  in self.sites.items():
            for op_name in site["operators"]:
                op_site = f"{op_name}@{site_name}"
                self.site_operator(op_site)

    
    def _load_global_ops(self):
        names = [name for name in self.basis.global_ops]
        for gop in names:
            self.global_operator(gop)
        
    def site_operator(self, name: str, site:str="") -> "ProductOperator":
        """
        Return a global operator representing an operator `name`
        acting over the site `site`. By default, the name is assumed
        to specify both the name and site in the form `"name@site"`.
        """
        if site !="":
            op_name = name
            name = f"{name}@{site}"
        else:
            op_name, site = name.split("@")

        op = self.site_operators.get(site, {}).get(name, None)
        if op is not None:
            return op

        local_op = self.sites[site]["operators"].get(op_name, None)
        if local_op is None:
            return None
        op = ProductOperator({site: local_op}, 1., self)
        self.site_operators.setdefault(site, {})
        self.site_operators[site][op_name] = op
        return op

    def bond_operator(self, name: str, src: str, dst: str, skip=None) -> "NBodyOperator":
        op = self.bond_operators.get( (name, src, dst,), None)
        if op is not None:
            return op
        # TODO: Take into account fermionic ops
        op = self.bond_operators.get( (name, dst, src,), None)
        if op is not None:
            return op
        # Try to build the bond operator from the descriptors.
        bond_op_descriptors = self.basis.bond_ops
        bond_op_descriptor = bond_op_descriptors.get(name, None)
        if bond_op_descriptor is None:
            return None
        bond_op_descriptor = bond_op_descriptor.replace("@", "__")
        src_operators =  self.sites[src]["operators"]
        dst_operators =  self.sites[dst]["operators"]
        
        # Load site operators on src and dst
        parms_and_ops = {f"{name_src}__src": self.site_operator(name_src, src)  for name_src in src_operators}        
        parms_and_ops.update({f"{name_dst}__dst": self.site_operator(name_dst, dst)  for name_dst in dst_operators})
        parms_and_ops.update(self.parms)


        # Try to evaluate
        result = eval_expr(bond_op_descriptor, parms_and_ops)
        if not (result is None or isinstance(result, str)):
            self.bond_operators[(name, src, dst,)] = result
            return result

        # Now, try to include existent bond operators
        parms_and_ops.update({f"{tup_op[0]}__src_dst": op for tup_op, op in self.bond_operators.items() if tup_op[0]==src and tup_op[1]==dst})
        parms_and_ops.update({f"{tup_op[0]}__dst_src": op for tup_op, op in self.bond_operators.items() if tup_op[0]==dst and tup_op[1]==src})

        result = eval_expr(bond_op_descriptor, parms_and_ops)
        if not(result is None or isinstance(result, str)):
            self.bond_operators[tuple(name, src, dst)] = result
            return result

        # Finally, try to load other operators
        if skip is None:
            skip = [name]
        else:
            skip.append(name)
        for bop in bond_op_descriptors:
            # Skip this name
            if bop in skip:
                continue
            # src->dst
            new_bond_op = self.bond_operator(bop, src, dst, skip)
            if new_bond_op is None:
                continue
            else:
                parms_and_ops[f"{bop}__src_dst"] = new_bond_op

            new_bond_op = self.bond_operator(bop, dst, src, skip)
            if new_bond_op is None:
                continue
            else:
                parms_and_ops[f"{bop}__dst_src"] = new_bond_op

            result = eval_expr(bond_op_descriptor, parms_and_ops)
            if result is not None and not isinstance(result, str):
                self.bond_operators[(name, src, dst,)] = result
                return result

        # If this fails after exploring all the operators, then it means that
        # the operator is not in the basis.
        if skip[-1]==name:
            self.bond_operators[(name, src, dst,)] = None
        return None

    def global_operator(self, name, skip=None):
        op = self.global_operators.get(name, None)
        if op is not None:
            return op
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
                if site_type is not None and site_type!=node_type:
                    continue
                site = self.sites[node_name]
                s_expr = expr.replace("#", node_type).replace("@", "__")
                s_parm = {key.replace("#", node_type):val for key, val in t_parm.items()}
                s_parm.update({f"{name_op}_local":op  for name_op, op in self.site_operators[node_name].items()})
                term_op = eval_expr(s_expr, s_parm)
                if term_op is None or isinstance(term_op, str):
                    # print("    ", name,"=", s_expr,[node_name]," does not match with the basis. remove it")
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
                    e_parms = { key.replace("#", f"{edge_type}"):val for key,val in t_parm.items()}
                    e_parms.update({ f"{name}__src": val for name, val in self.site_operators[src].items()})
                    e_parms.update({ f"{name}__dst": val for name, val in self.site_operators[dst].items()})
                    # Try to compute using only site terms
                    term_op = eval_expr(e_expr, e_parms)
                    if not isinstance(term_op, str):
                        bond_terms.append(term_op)
                        continue
                    # Try now adding the bond operators
                    for name_bop in self.basis.bond_ops:
                        self.bond_operator(name_bop, src, dst)
                        self.bond_operator(name_bop, dst, src)
                    
                    e_parms.update({ f"{key[0]}__src_dst":val  for key, val in self.bond_operators.items() if key[1]==src and key[2]==dst})
                    e_parms.update({ f"{key[0]}__dst_src":val  for key, val in self.bond_operators.items() if key[2]==src and key[1]==dst})
                    term_op = eval_expr(e_expr, e_parms)
                    if not isinstance(term_op, str):
                        bond_terms.append(term_op)
                    else:
                        # print("    ", name,"=", e_expr, [src,dst]," does not match with the basis. remove it")
                        self.basis.global_ops.pop(name)
                        return None

        result = sum(site_terms) + sum(bond_terms)
        self.global_operators[name] = result
        return result



class Operator:
    def __sub__(self, op):
        nop = - op
        return self + nop

    def __radd__(self, op):
        return self + op
        
    def __rsub__(self, op):
        nop = - self
        return op + nop
    

        
class ProductOperator(Operator):
    def __init__(self, sites_op: dict, prefactor=1.0, system=None):
        self.sites_op = sites_op
        self.prefactor = prefactor
        self.system = system
        if system is not None:
            self.size = len(system.sites)
            self.dimensions = {name: site["dimension"]  for name, site in  system.sites.items()}


    def __add__(self, op):
        if self.prefactor == 0:
            return op
        if isinstance(op, (int, float, complex)):
            if op == 0:
                return self
            op = ProductOperator({}, op, self.system)
        elif isinstance(op, ProductOperator):
            if op.prefactor == 0:
                return self
            if len(op.sites_op) == 1 and len(self.sites_op) == 1:
                site = [k for k in self.sites_op][0]
                if site in op.sites_op:
                    return ProductOperator(
                        {
                            site: (
                                self.sites_op[site] * self.prefactor
                                + op.prefactor * op.sites_op[site]
                            )
                        },
                        prefactor=op.prefactor * self.prefactor,
                        system=self.system,
                    )
            new_terms = [op, self]
        elif isinstance(op, NBodyOperator):
            new_terms = op.terms + [self]    
        return NBodyOperator(new_terms)

    def __mul__(self, op):
        if isinstance(op, (int, float, complex)):
            new_prefactor = self.prefactor * op
            if new_prefactor == 0.0:
                return ProductOperator({}, prefactor=new_prefactor, system=self.system)
            return ProductOperator(
                self.sites_op, prefactor=new_prefactor, system=self.system
            )
        if isinstance(op, ProductOperator):
            new_sites_op = self.sites_op
            for pos, factor in op.sites_op.items():
                if pos in new_sites_op:
                    new_sites_op[pos] = new_sites_op[pos] * factor
                else:
                    new_sites_op[pos] = factor
            return ProductOperator(
                new_sites_op,
                prefactor=self.prefactor * op.prefactor,
                system=self.system,
            )
        if isinstance(op, NBodyOperator):
            new_terms = [self * op2 for op2 in op.terms_coeffs]
            new_terms = [op for op in new_terms if op.prefactor]
            return NBodyOperator(new_terms)

        raise NonImplementedError

    def __neg__(self):
        return ProductOperator(self.sites_op, -self.prefactor, self.system)
    
    def __rmul__(self, op):
        if isinstance(op, (int, float, complex)):
            return self * op
        return NonImplementedError

    def to_qutip(self):
        if self.prefactor == 0:
            return self.prefactor
        if len(self.sites_op) == 0:
            return self.prefactor
        ops = self.sites_op
        return self.prefactor * qutip.tensor(
            [
                ops[site] if site in ops else qutip.qeye(dim)
                for site, dim in self.system.dimensions.items()
            ]
        )


class NBodyOperator(Operator):
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

                dimensions = system.dimensions

    def __add__(self, op):
        if isinstance(op, (int, float, complex)):
            if op==0.:
                return self
            op = ProductOperator({}, op, self.system)
            
        elif isinstance(op, ProductOperator):
            if op.prefactor == 0:
                return self
            new_terms = self.terms + [op]
        elif isinstance(op, NBodyOperator):
            if len(op.terms) == len(self.terms) == 1:
                return self.terms[0] + op.terms[0]
            new_terms = self.terms + op.terms
        # TODO: cancel terms
        return NBodyOperator(new_terms)

    def __rmul__(self, op):
        if isinstance(op, (int, float, complex)):
            return self * op
        return NonImplementedError

    def __mul__(self, op):
        if isinstance(op, (int, float, complex)):
            if op == 0:
                return op
            new_terms = [op * op1 for op1 in self.terms if op1.prefactor]
        elif isinstance(op, ProductOperator):
            if op.prefactor:
                new_terms = [op1 * op for op1 in self.terms if op1.prefactor]
            else:
                new_terms = []
        elif isinstance(op, NBodyOperator):
            return NBodyOperator(
                [op1 * op2 for op1 in self.terms for op2 in op.terms]
            )

        if len(new_terms) == 0:
            return 0.0
        if len(new_terms) == 1:
            return new_terms[0]
        return NBodyOperator(new_terms)

    def to_qutip(self):
        if len(self.terms) == 0:
            return 0
        return sum(t.to_qutip() for t in self.terms)
