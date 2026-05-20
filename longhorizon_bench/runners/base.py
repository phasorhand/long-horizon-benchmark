# longhorizon_bench/runners/base.py
"""Abstract base runner for LongHorizon-Bench."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from longhorizon_bench.schema import AgentAction, CheckpointResponse
from longhorizon_bench.env import LongHorizonEnv


class BaseRunner(ABC):
    @abstractmethod
    def act(self, observation: dict[str, Any]) -> AgentAction | CheckpointResponse:
        ...

    def run(self, env: LongHorizonEnv) -> dict[str, Any]:
        obs = env.reset()
        done = False
        while not done:
            action = self.act(obs)
            obs, reward, done, info = env.step(action)
        return env.evaluate()
