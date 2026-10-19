"""Public entry points for the isolated ACP S1--S3 experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .alternative_env import AlternativeMSWEnv, canonical_archetype
from .alternative_rewards import ACP_TO_REWARD_MODE, RewardSpec

METRIC_VERSION = "mee-net-products-v2"
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PACKAGE_ROOT / "data" / "train_data.xlsx"
REGISTRY_FILE = PACKAGE_ROOT / "data" / "alternative_product_registry.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(archetype, data_file=None):
    """Reject a different training matrix rather than join unverified recipes."""
    data_file = Path(data_file or DEFAULT_DATA)
    registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    source_hash = registry["source"]["sha256"]
    if sha256_file(data_file) != source_hash:
        raise ValueError("Training data SHA256 does not match the validated product registry")
    arch = canonical_archetype(archetype)
    raw = registry["archetypes"][arch]
    statuses = raw["metric_status"]
    return {
        "version": METRIC_VERSION,
        "archetype": arch,
        "source_sha256": source_hash,
        "data_sha256": source_hash,
        "registry_sha256": sha256_file(REGISTRY_FILE),
        "products": {key: {**metadata, **raw["products"][key]}
                     for key, metadata in registry["products"].items()},
        "routes": {key: route["products"] for key, route in raw["routes"].items()},
        "gross_cost_contract": registry["gross_cost_contract"]["archetypes"][arch],
        "metric_status": {
            "net_cost": "verified" if statuses["S1"] == "verified" else statuses["S1"],
            "net_energy": "verified" if statuses["S2"] == "verified_product_output" else statuses["S2"],
            "resource_output": "verified" if statuses["S3"] == "verified_product_output" else statuses["S3"],
        },
        "recipe_source_sheet": raw["recipe_source_sheet"],
        "price_source": raw.get("price_source"),
        "method": registry["metric_contract"],
        "known_limitations": registry.get("issues", []) + registry["gross_cost_contract"].get("issues", []),
    }


def make_alternative_env(archetype, acp, scale, data_file=None):
    if acp not in ACP_TO_REWARD_MODE:
        raise ValueError(f"Unknown new ACP label: {acp}")
    data_file = Path(data_file or DEFAULT_DATA)
    contract = load_contract(archetype, data_file)
    spec = RewardSpec(ACP_TO_REWARD_MODE[acp], float(scale), METRIC_VERSION)
    env = AlternativeMSWEnv(archetype, data_file, spec, contract)
    env.data_file = str(data_file.resolve())
    return env
