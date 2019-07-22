import logging
from typing import Tuple

import numpy as np


class SolverInterface:
    logger = logging.getLogger(__name__)

    def solve(self, signal: np.ndarray, *args) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
