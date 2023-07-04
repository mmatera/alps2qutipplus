from typing import Optional
import numpy as np

from alpsqutip.model import SystemDescriptor, NBodyOperator, Operator, ProductOperator


class DensityOperatorMixin:
    def expect(self, obs):
        return (self * obs).tr()


class ProductDensityOperator(ProductOperator, DensityOperatorMixin):
    def __init__(
        self,
        local_states: dict,
        weight: float = 1.0,
        system: Optional[SystemDescriptor] = None,
        normalize: bool = True,
    ):
        assert weight >= 0
        sites = tuple(system.sites.keys() if system else local_states.keys())
        dimensions = system.dimensions
        zs = {
            site: (local_states[site].tr()
            if site in local_states
            else dimensions[site])
            for site in sites
        }

        if normalize:
            assert (z > 0 for z in zs.values())
            local_states = {site:
                sigma / zs[site] for site, sigma in local_states.items()
            }
            for z in zs.values():
                weight /= z

        super().__init__(local_states, prefactor=weight, system=system)
        self.fs = {site: -np.log(z) for site, z in zs.items()}

    def expect(self, obs: Operator):
        sites_obs = obs.sites_op
        local_states = self.sites_op
        factors = [
            (local_states[site] * op).tr() for site, op in sites_obs.items()
        ]
        result = 1
        for f in factors:
            result *= f
        return result

    def partial_trace(self, sites: list):
        sites_op = self.sites_op
        sites_in = [site for site in sites if site in sites_op]
        local_states = {site: sites_op[site] for site in sites_in}
        subsystem = self.system.subsystem(sites_in)
        return ProductDensityOperator(
            local_states, self.prefactor, subsystem, normalize=False
        )

    def __add__(self, rho: Operator):
        system = self.system
        if isinstance(rho, float):
            sites_op = self.sites_op
            n_factors = len(sites_op)
            if n_factors == 0:
                return ProductDensityOperator(
                    {}, self.prefactor + rho, self.system, False
                )
            if n_factors == 1:
                site, sigma = next(sites_op.items())
                local_dim = (
                    1 / system.dimensions.get(site, 1) if system else 1.0
                )
                prefactor = self.prefactor
                new_prefactor = prefactor + rho / local_dim
                return ProductDensityOperator(
                    {
                        site: prefactor * sigma + rho / local_dim,
                    },
                    new_prefactor,
                    system,
                )
            return SeparableDensityOperator(
                [ProductDensityOperator({}, rho, self.system, False)]
            )

        if isinstance(rho, ProductDensityOperator):
            sites_op = self.sites_op
            n_factors = len(sites_op)
            other_sites_op = rho.sites_op
            other_n_factors = len(other_sites_op)
            if len(sites_op) == 0:
                return rho + self.prefactor
            if len(other_sites_op) == 0:
                return self + rho.prefactor

            if 1 == n_factors == other_n_factors:
                site, sigma = next(sites_op.items())
                other_sigma = other_sites_op.get(site, None)
                if other_sigma is not None:
                    a, b = self.prefactor, rho.prefactor
                    new_prefactor = a + b
                    a, b = a / new_prefactor, b / new_prefactor
                    new_sigma = a * sigma + b * other_sigma
                    return ProductDensityOperator(
                        {
                            site: new_sigma,
                        },
                        new_prefactor,
                        self.system,
                        False,
                    )
            return SeparableDensityOperator([self, rho])
        if isinstance(rho, SeparableDensityOperator):
            return SeparableDensityOperator(rho.terms + [self])
        return super().__add__(rho)

    def __mul__(self, a):
        if isinstance(a, float):
            if a > 0:
                return SeparableDensityOperator(
                    [
                        ProductDensityOperator(
                            t.sites_op, t.prefactor * a, t.system, False
                        )
                        for t in self.terms
                    ]
                )
            if a == 0.0:
                return ProductDensityOperator({}, a, self.system, False)
        return super().__mul__(a)


class SeparableDensityOperator(NBodyOperator, DensityOperatorMixin):
    def __init__(self, terms: list):
        assert (isinstance(t, ProductDensityOperator) for t in terms)
        super().__init__(terms)

    def expect(self, obs: Operator):
        normalization = sum(t.prefactor for t in self.terms)
        return (
            sum(t.expect(obs) * t.prefactor for t in self.terms)
            / normalization
        )

    def partial_trace(self, sites: list):
        return SeparableDensityOperator(
            [t.partial_trace(sites) for t in self.terms]
        )

    def __add__(self, rho: Operator):
        if isinstance(rho, SeparableDensityOperator):
            return SeparableDensityOperator(self.terms + rho.terms)
        if isinstance(rho, ProductDensityOperator):
            return SeparableDensityOperator
        (self.terms + [rho])
        if isinstance(rho, float):
            return SeparableDensityOperator(
                self.terms
                + [
                    ProductDensityOperator(
                        {}, rho, self.terms[0].system, False
                    )
                ]
            )
        return super().__add__(rho)

    def __mul__(self, a):
        if isinstance(a, float):
            return SeparableDensityOperator(
                [
                    ProductDensityOperator(
                        t.sites_op, t.prefactor * a, t.system, False
                    )
                    for t in self.terms
                ]
            )
        return super().__mul__(a)


"""
class QuantumState:
    def expect(self, op):
        if isinstance(op, (int, float, complex)):
            return op
        if isinstance(op, NBodyOperator):
            return sum([self.expect(t) for t in op.terms])
        if isinstance(op, ProductOperator):
            return self._compute_expect_over_product_operator(op)
        raise NotImplementedError


class ProductQuantumState(QuantumState, ProductOperator):
    def __init__(
        self, local_states, system, weight=1, normalize_local_states=True
    ):
        super(ProductQuantumState, self).__init__(
            sites_op=local_states, prefactor=1, system=system
        )
        self.weight = weight
        # Normalize local states and adjust the prefactor
        local_states = self.sites_op
        normalization = reduce(
            lambda x, y: x * y,
            (d for s, d in system.dimensions.items() if s not in local_states),
            1.0,
        )

        if normalize_local_states:
            for s, op in local_states.items():
                local_trace = op.tr()
                if local_trace <= 0:
                    raise ValueError(f"Non-positive trace on site {s}")
                local_states[s] = op / local_trace

        self.prefactor = weight / normalization

    def _compute_expect_over_product_operator(self, op):
        dimensions = self.system.dimensions
        site_ops = op.sites_op
        site_states = self.sites_op
        factors = [
            (site_ops[s] * site_states[s]).tr()
            if s in site_states
            else site_ops[s].tr() / dimensions[s]
            for s in site_ops
        ]
        return reduce(lambda x, y: x * y, factors, op.prefactor)

    def partial_trace(self, sites):
        sites_op = self.sites_op
        sites = [s for s in sites if s in self.system.sites]
        system = self.system.subsystem(sites)
        sites_op = {s: sites_op[s] for s in sites if s in sites_op}
        return ProductQuantumState(
            sites_op,
            system=system,
            weight=self.weight,
            normalize_local_states=False,
        )

    def __add__(self, op):
        system = self.system
        if isinstance(op, (int, float)):
            if op == 0:
                return self
            if op > 0:
                rho = ProductQuantumState({}, system, weight=op)
                return MixedQuantumState([self, rho])
        elif isinstance(op, ProductQuantumState):
            if op.prefactor == 0:
                return self
            return MixedQuantumState([self, op])
        elif isinstance(op, MixedQuantumState):
            return MixedQuantumState([self] + op.terms)
        return super(ProductQuantumState, self).__add__(op)

    def __mul__(self, op):
        system = self.system
        if isinstance(op, (int, float)) and op >= 0:
            rho = ProductQuantumState(
                self.sites_op.copy(), system, weight=op * self.weight
            )
            return rho

        return super(ProductQuantumState, self).__mul__(op)


class MixedQuantumState(QuantumState, NBodyOperator):
    def __init__(self, terms):
        super(MixedQuantumState, self).__init__(terms)

    def __add__(self, op):
        system = self.system
        if isinstance(op, (int, float)):
            if op > 0:
                rho = ProductQuantumState({}, system, weight=op)
                return MixedQuantumState(self.terms + [rho])
            elif op == 0:
                return self
        elif isinstance(op, ProductQuantumState):
            if op.prefactor == 0:
                return self
            return MixedQuantumState(self.terms + [op])
        elif isinstance(op, MixedQuantumState):
            return MixedQuantumState(self.terms + op.terms)
        return super(MixedQuantumState, self).__add__(op)

    def __mul__(self, op):
        if isinstance(op, (int, float)) and op >= 0:
            return MixedQuantumState([t * op for t in self.terms])

        return super(MixedQuantumState, self).__mul__(op)

    def _compute_expect_over_product_operator(self, op):
        return sum(
            t.weight * t._compute_expect_over_product_operator(op)
            for t in self.terms
        )

    def partial_trace(self, sites: list):
        return sum(t.partial_trace(sites) for t in self.terms)
"""
