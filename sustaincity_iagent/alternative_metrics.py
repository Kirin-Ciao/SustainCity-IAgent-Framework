"""Strict original MEE route accounting plus versioned net product metrics."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping
import numpy as np

from .alternative_rewards import MetricUnavailableError

COMPOSITION_KEYS = ("food", "glass", "metal", "paper", "plastic", "rubber", "wood", "yard", "other")
ROUTE_GROUPS = (
    (3, 10, "food", ("food_incineration", "food_landfill", "food_AD", "food_composting", "food_other_biotech", "food_D&AF", "food_sewer")),
    (10, 12, "glass", ("glass_recycle", "glass_landfill")),
    (12, 14, "metal", ("metal_recycle", "metal_landfill")),
    (14, 17, "paper", ("paper_recycle", "paper_incineration", "paper_landfill")),
    (17, 20, "plastic", ("plastic_recycle", "plastic_incineration", "plastic_landfill")),
    (20, 23, "rubber", ("rubber_recycle", "rubber_incineration", "rubber_landfill")),
    (23, 26, "wood", ("wood_recycle", "wood_incineration", "wood_landfill")),
    (26, 29, "yard", ("yard_composting", "yard_incineration", "yard_landfill")),
)


def route_masses(ip, composition: Mapping[str, float], total_mass: float) -> dict[str, float]:
    """Original controlled/unsorted/classified routes; independent conservation check."""
    ip = np.asarray(ip, dtype=np.float64)
    if ip.shape != (29,) or not np.isfinite(ip).all() or (ip < 0).any() or (ip > 1).any():
        raise ValueError("Invalid 29-IP configuration")
    mass = float(total_mass)
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("Functional-unit mass must be finite and positive")
    shares = np.asarray([composition[k] for k in COMPOSITION_KEYS], dtype=float)
    if not np.isfinite(shares).all() or (shares < 0).any() or abs(shares.sum() - 1) > 1e-6:
        raise ValueError("Composition must be a finite nonnegative simplex")
    controlled = mass * ip[0]
    classified = controlled * ip[1]
    unclassified = controlled * (1 - ip[1])
    routes = {
        "open_burning": (mass - controlled) * .3,
        "open_dumping": (mass - controlled) * .7,
        "unsorted_incineration": unclassified * ip[2],
        "unsorted_landfill": unclassified * (1 - ip[2]),
    }
    for start, end, waste, methods in ROUTE_GROUPS:
        if abs(ip[start:end].sum() - 1) > 1e-6:
            raise ValueError(f"Non-simplex {waste} routing")
        for method, fraction in zip(methods, ip[start:end]):
            routes[method] = classified * composition[waste] * fraction
    routes["other_landfill"] = classified * composition["other"]
    residual = sum(routes.values()) - mass
    if abs(residual) > 1e-6 * mass or min(routes.values()) < -1e-10:
        raise FloatingPointError(f"Primary route mass conservation failed: {residual}")
    return routes


def project_action(previous_ip, action) -> np.ndarray:
    """Match original clipping/simplex operation for all finite inputs."""
    previous_ip = np.asarray(previous_ip, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (29,) or not np.isfinite(action).all():
        raise ValueError("Action must be 29 finite continuous increments")
    ip = np.clip(previous_ip + np.clip(action, -.2, .2), 0., 1.)
    for start, end, *_ in ROUTE_GROUPS:
        subtotal = np.sum(ip[start:end])
        if subtotal < 1e-8:
            ip[start:end] = 0
            ip[start] = 1
        else:
            ip[start:end] /= subtotal
    if not np.isfinite(ip).all():
        raise FloatingPointError("Nonfinite route projection")
    return ip.astype(np.float32)


@dataclass
class MetricCalculator:
    engine: object
    contract: Mapping

    def __post_init__(self):
        # Never let the legacy step's nan_to_num hide an unknown coefficient.
        for route, params in self.engine.disposal_params.items():
            for key, value in params.items():
                if not math.isfinite(float(value)):
                    raise ValueError(f"Missing/nonfinite source coefficient: {route}.{key}")
        for key, value in self.engine.transport_params.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"Missing/nonfinite transport coefficient: {key}")

    def evaluate(self, ip) -> dict:
        mass = float(self.engine.total_mass)
        routes = route_masses(ip, self.engine.composition, mass)
        raw = [float(v) for v in self.engine.compute_metrics(np.asarray(ip, dtype=np.float32))]
        if len(raw) != 5 or not np.isfinite(raw).all():
            raise FloatingPointError(f"Nonfinite raw original MEE outputs: {raw}")
        ghg, energy_input, legacy_energy_output, gross_cost, legacy_resource = [v / mass for v in raw]
        products = {}
        product_routes = self.contract.get("routes", {})
        metadata = self.contract.get("products", {})
        cost_contract = self.contract.get("gross_cost_contract")
        legacy_gross_cost = gross_cost
        if cost_contract is None:
            raise MetricUnavailableError("Missing validated gross-cost accounting contract")
        unit_cost = cost_contract.get("corrected_total_route_cost_2020USD_per_t")
        gross_cost = None
        if unit_cost is not None:
            if set(unit_cost) != set(routes):
                raise MetricUnavailableError("Incomplete gross-cost route coverage")
            for route, value in unit_cost.items():
                if not math.isfinite(float(value)) or float(value) < 0:
                    raise ValueError(f"Missing/nonfinite/negative gross-cost coefficient: {route}")
            gross_cost = math.fsum(route_mass / mass * float(unit_cost[route])
                                   for route, route_mass in routes.items())
        elif not cost_contract.get("status", "").startswith("blocked"):
            raise MetricUnavailableError("Verified gross-cost contract lacks coefficients")
        for route, route_mass in routes.items():
            if route not in product_routes:
                if metadata:
                    raise MetricUnavailableError(f"Missing product-yield route: {route}")
                continue
            if set(product_routes[route]) != set(metadata):
                raise MetricUnavailableError(f"Incomplete product-yield coverage: {route}")
            for product, coefficient in product_routes[route].items():
                coefficient = float(coefficient)
                if not math.isfinite(coefficient) or coefficient < 0:
                    raise ValueError(f"Invalid product coefficient: {route}.{product}")
                products[product] = products.get(product, 0.) + route_mass / mass * coefficient
        status = self.contract.get("metric_status", {})
        net_cost = resource = benefits = energy_output = net_energy = None
        if status.get("net_energy") == "verified":
            energy_output = math.fsum(
                amount * float(metadata[product]["energy_mj_per_unit"])
                for product, amount in products.items()
            )
            for product in products:
                factor = float(metadata[product]["energy_mj_per_unit"])
                if not math.isfinite(factor) or factor < 0:
                    raise ValueError(f"Invalid physical-energy conversion: {product}")
            net_energy = energy_input - energy_output
        if status.get("net_cost") == "verified":
            if gross_cost is None:
                raise MetricUnavailableError("Verified net cost requires a verified gross cost")
            revenue = []
            for product, amount in products.items():
                info = metadata[product]
                price = info.get("price_usd2020_per_unit")
                if price is None or not math.isfinite(float(price)) or float(price) < 0:
                    raise MetricUnavailableError(f"Missing/invalid USD2020 price for {product}")
                revenue.append(amount * float(price))
            benefits = math.fsum(revenue)
            net_cost = gross_cost - benefits
        if status.get("resource_output") == "verified":
            amounts = []
            for product, amount in products.items():
                if metadata[product].get("is_resource", False):
                    if metadata[product]["unit"] != "kg":
                        raise ValueError(f"Resource product is not a physical kg quantity: {product}")
                    amounts.append(amount)
            resource = math.fsum(amounts)
        result = {
            "ghg": ghg, "gross_cost": gross_cost, "economic_benefits": benefits,
            "net_cost": net_cost, "energy_input": energy_input, "energy_output": energy_output,
            "net_energy": net_energy, "resource_output": resource,
            "legacy_energy_output_aggregate": legacy_energy_output,
            "legacy_gross_cost_aggregate": legacy_gross_cost,
            "legacy_resource_aggregate": legacy_resource, "products": products,
            "route_mass_per_tonne": {k: v / mass for k, v in routes.items()},
            "primary_mass_relative_residual": abs(sum(routes.values()) - mass) / mass,
            "metric_version": self.contract.get("version", "energy-engine-v1"),
            "metric_status": dict(status),
            "gross_cost_status": cost_contract["status"],
            "units": {"ghg": "kgCO2e/tMSW", "net_cost": "USD2020/tMSW",
                      "net_energy": "MJ/tMSW", "resource_output": "kg products/tMSW"},
        }
        for key in ("ghg", "gross_cost", "economic_benefits", "net_cost", "energy_input", "energy_output", "net_energy", "resource_output"):
            if result[key] is not None and not math.isfinite(result[key]):
                raise FloatingPointError(f"Nonfinite physical metric: {key}")
        return result
