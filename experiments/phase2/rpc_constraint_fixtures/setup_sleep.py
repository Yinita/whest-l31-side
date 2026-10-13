"""Harness fault fixture only; never submit this file."""
import time

import flopscope.numpy as fnp
from whestbench import BaseEstimator


class Estimator(BaseEstimator):
    def setup(self, context):
        time.sleep(5.2)

    def predict(self, mlp, budget):
        return fnp.zeros((16, 1024), dtype=fnp.float32)
