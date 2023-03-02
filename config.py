from dataclasses import dataclass

@dataclass
class Params:
    max_epochs: int
    lr: float
    batch_size: int
    optimizer: str

@dataclass
class FMNISTConfig:
    params: Params