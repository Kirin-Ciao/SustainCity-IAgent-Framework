"""
SustainCity-IAgent Framework

High-level API for training and applying intelligent agents
for municipal solid waste (MSW) management scenarios.
"""

from .train import train_msw_agent
from .predict import predict_msw_scenario

__all__ = ["train_msw_agent", "predict_msw_scenario"]