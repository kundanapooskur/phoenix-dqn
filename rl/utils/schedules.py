import math
from dataclasses import dataclass


@dataclass
class EpsilonSchedule:
    start: float = 1.0
    end: float = 0.01
    decay_steps: int = 1_000_000

    def value(self, step: int) -> float:
        if step >= self.decay_steps:
            return self.end
        fraction = step / float(self.decay_steps)
        return self.start + fraction * (self.end - self.start)


@dataclass
class ExponentialEpsilon:
    start: float = 1.0
    end: float = 0.01
    decay_rate: float = 0.999_5

    def value(self, step: int) -> float:
        return max(self.end, self.start * (self.decay_rate ** step))


class BoltzmannExplorer:
    """Softmax (Boltzmann) exploration temperature schedule."""

    def __init__(self, start_temperature: float = 1.0, end_temperature: float = 0.1, decay_steps: int = 500_000):
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature
        self.decay_steps = decay_steps

    def temperature(self, step: int) -> float:
        if step >= self.decay_steps:
            return self.end_temperature
        fraction = step / float(self.decay_steps)
        return self.start_temperature + fraction * (self.end_temperature - self.start_temperature)



