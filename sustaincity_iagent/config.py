"""
Configuration constants for the SustainCity-IAgent framework.
"""

IAS_TO_REWARD_MODE = {
    "Scenario I": "carbon",
    "Scenario II": "carbon_cost",
    "Scenario III": "carbon_energy",
    "Scenario IV": "carbon_resource",
    "Scenario V": "carbon_biomass_led",
    "Scenario VI": "carbon_incineration_led",
    "Scenario VII": "ssp3_weighted_carbon",
    "Scenario VIII": "ssp4_weighted_multi",
    "Scenario IX": "ssp5_cost_service_led",
}

REWARD_MODE_TO_IAS = {v: k for k, v in IAS_TO_REWARD_MODE.items()}

SUPPORTED_ARCHETYPES = {"JPN", "EU", "US", "CHN", "IND", "Permissive"}

SUPPORTED_SSP = ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]

DEFAULT_TRAIN_DATA_FILE = "data/train_data.xlsx"
DEFAULT_COUNTRY_DATA_FILE = "data/country_data.xlsx"
