from dataclasses import dataclass
from typing import List, Literal

from util.hparams import HyperParams


@dataclass
class originalHyperParams(HyperParams):
    # Method
    model_name: str

