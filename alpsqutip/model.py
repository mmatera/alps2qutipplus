import qutip

from alpsqutip.geometry import GraphDescriptor


class SystemDescriptor:
    def __init__(self, graph: GraphDescriptor, basis: dict, parms: dict):
        self.graph = graph
        self.basis = basis
        self.parms = basis.parms
        self.parms.update(parms)
        self.sites = {
            node: basis.site_basis[attr["type"]] for node, attr in graph.nodes.items()
        }
        self.site_operators = {}
        self.bond_operators = {}
        self.global_operators = {}
        self._load_site_operators()

    def _load_site_operators():
        for site in self.sites:
            for op_name in site.operators:
                site_operator("{op_name}@{site}")
        print(self.site_operators.keys())

    def site_operator(self, name: str) -> ProductOperator:
        op = self.site_operators.get(name)
        if op is not None:
            return op

        op_name, site = name.split("@")
        local_op = self.sites.operators.get(op_name, None)
        if local_op is None:
            return None
        op = ProductOperator({site: local_op}, self)
        self.site_operators[name] = op
        return op

    def bond_operator(self, name: str, src: str, tgt: str) -> NBodyOperator:
        op = self.bond_operators.get(name, None)
        if op is not None:
            return op
        # Build the global_operator from the descriptor
        name, site = name.split("@")
        op_descr = self.basis.bond_operators.get(name + "_src_dst", None)
        if op_descr is None:
            # this operator is not a defined bond_operator
            return None

    def global_operator(self, name):
        op = self.global_operators.get(name, None)
        if op is not None:
            return op
        # Build the global_operator from the descriptor
        op_descr = self.basis.global_operators.get(name, None)


class ProductOperator(Operator):
    def __init__(self, sites_op: dict, prefactor=1.0, system=None):
        self.sites_op = sites_op
        self.prefactor = prefactor
        size = max(sites_op.keys()) + 1 if len(sites_op) else 0
        if system is None:
            dimensions = [
                reduce(lambda x, y: x * y, sites_op[s].dims[0]) if s in sites_op else 1
                for s in range(size)
            ]
            self.system = SystemDescriptor(dimensions)
        else:
            curr_size = len(system.dimensions)
            if size > curr_size:
                system.dimensions = system.dimensions + (size - curr_size) * [1]
            for s in sites_op:
                s_dim = system.dimensions[s]
                if s_dim != sites_op[s].dims[0][0]:
                    if s_dim == 1:
                        system.dimensions[s] = sites_op[s].dims[0][0]
                    else:
                        raise Exception(
                            f"wrong dimension for {s}: {s_dim}!={sites_op[s].dims[0][0]}",
                        )
            self.system = system

    def __add__(self, op):
        if self.prefactor == 0:
            return op
        if op.prefactor == 0:
            return self
        if isinstance(op, NBodyOperator):
            new_terms = op.terms + [self]
        elif isinstance(op, ProductOperator):
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
                ops[i] if i in ops else qutip.qeye(dim)
                for i, dim in enumerate(self.system.dimensions)
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
        if isinstance(op, ProductOperator):
            if op.prefactor == 0:
                return self
            new_terms = self.terms + [op]
        if isinstance(op, NBodyOperator):
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
            new_terms = [op * op1 for op1 in self.terms_coeffs if op1.prefactor]
        elif isinstance(op, ProductOperator):
            if op.prefactor:
                new_terms = [op1 * op for op1 in self.terms_coeffs if op1.prefactor]
            else:
                new_terms = []
        elif isinstance(op, NBodyOperator):
            return NBodyOperator(
                [op1 * op2 for op1 in self.terms_coeffs for op2 in op.terms_coeffs]
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
