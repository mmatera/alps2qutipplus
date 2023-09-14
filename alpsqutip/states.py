"""
Density operator classes
"""
from functools import reduce
from typing import Optional
import numpy as np

from alpsqutip.model import SystemDescriptor, Operator
from alpsqutip.operators import LocalOperator, OneBodyOperator, ProductOperator, SumOperator


class DensityOperatorMixin:
    """
    DensityOperatorMixin is a Mixing class that
    contributes operator subclasses with the method
    `expect`.
    """

    def expect(self, obs):
        """Compute the expectation value of an observable"""
        return (self * obs).tr()


class ProductDensityOperator(ProductOperator, DensityOperatorMixin):
    """An uncorrelated density operator"""

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
        local_zs = {
            site: (local_states[site].tr()
                   if site in local_states
                   else dimensions[site])
            for site in sites
        }

        if normalize:
            assert (z > 0 for z in local_zs.values())
            local_states = {site:
                            sigma / local_zs[site] for site, sigma in local_states.items()
                            }
            for local_z in local_zs.values():
                weight /= local_z

        super().__init__(local_states, prefactor=weight, system=system)
        self.local_fs = {site: -np.log(z) for site, z in local_zs.items()}

    def expect(self, obs: Operator):
        sites_obs = obs.sites_op
        local_states = self.sites_op
        factors = [
            (local_states[site] * op).tr() for site, op in sites_obs.items()
        ]
        result = 1
        for local_f in factors:
            result *= local_f
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
                    a_float, b_float = self.prefactor, rho.prefactor
                    new_prefactor = a_float + b_float
                    a_float, b_float = a_float / new_prefactor, b_float / new_prefactor
                    new_sigma = a_float * sigma + b_float * other_sigma
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
                return ProductDensityOperator(
                    self.sites_op, self.prefactor * a, self.system, False
                )

            if a == 0.0:
                return ProductDensityOperator({}, a, self.system, False)
        return super().__mul__(a)


class SeparableDensityOperator(SumOperator, DensityOperatorMixin):
    """
    A mixture of product operators
    """

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
            return SeparableDensityOperator(self.terms + [rho])
        if isinstance(rho, float):
            return SeparableDensityOperator(
                self.terms + [
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


class GibbsDensityOperator(Operator, DensityOperatorMixin):
    """
    stores an operator of the form rho=exp(-K)/Tr(exp(-K))
    """

    def __init__(self, K: Operator, system: SystemDescriptor = None):
        self.K = K
        self.system = system or K.system

    def __neg__(self):
        return -self.to_qutip_operator()

    def partial_trace(self, sites):
        return self.to_qutip_operator().partial_trace(sites)

    def to_qutip(self):
        # TODO: add offset to the lowest eigenvalue
        rho_qutip = (-self.K).to_qutip().expm()
        return rho_qutip / rho_qutip.tr()


class GibbsProductDensityOperator(Operator, DensityOperatorMixin):
    """
    stores an operator of the form rho=\\otimes_i exp(-K_i)/Tr(exp(-K_i))
    """

    def __init__(self, K: Operator, system: SystemDescriptor = None, normalize=True):
        if isinstance(K, LocalOperator):
            if normalize:
                self.k_by_site = {K.site: K.operator+np.log(K.expm().tr())}
            else:
                self.k_by_site = {K.site: K.operator}
        elif isinstance(K, OneBodyOperator):
            k_by_site = {k_local.site: k_local.operator for k_local in K.terms}
            if normalize:
                k_by_site = {
                    site: local_k + np.log(K.expm().tr()) for site, local_k in k_by_site.values()}
            self.k_by_syte = k_by_site
        else:
            raise ValueError

        self.system = system or K.system

    def expect(self, operator):
        # TODO: write a better implementation
        return self.to_product_state().expect(operator)

    def partial_trace(self, sites):
        sites = [site for site in sites if site in self.system.dimensions]
        subsystem = self.system.subsystem(sites)
        return GibbsProductDensityOperator({self.k_by_site[site] for site in sites},
                                           subsystem, False)

    def to_product_state(self):
        """Convert the operator in a productstate"""

        dimensions = self.system.dimensions
        global_z = reduce(
            lambda x, y: x*y, [dim for site, dim in dimensions.values() if site not in self.k_by_site])

        return ProductDensityOperator({site: (-local_k).expm() for site, local_k in self.k_by_site.values()},
                                      1./global_z, system=self.system, normalize=False)

    def __neg__(self):
        return -self.to_product_state()

    def to_qutip(self):
        # TODO: add offset to the lowest eigenvalue
        return self.to_product_state().to_qutip()
