"""Harness fault fixture only; never submit this file."""
import time

import flopscope.numpy as fnp
from whestbench import BaseEstimator


class Estimator(BaseEstimator):
    def predict(self, mlp, budget):
        time.sleep(0.45)
        return fnp.zeros((16, 1024), dtype=fnp.float32)
