"""Harness fixture: preserve the first failure and recover the next network."""
import time

import flopscope.numpy as fnp
from whestbench import BaseEstimator


class Estimator(BaseEstimator):
    def predict(self, mlp, budget):
        if mlp.seed == 3909487726:
            time.sleep(0.45)
        return fnp.zeros((16, 1024), dtype=fnp.float32)
