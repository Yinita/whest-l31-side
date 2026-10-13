"""Real legal server work: wall > .4s must not imply residual failure."""
import flopscope.numpy as fnp
from whestbench import BaseEstimator


class Estimator(BaseEstimator):
    def predict(self, mlp, budget):
        matrix = fnp.ones((2048, 2048), dtype=fnp.float32)
        for _ in range(12):
            product = matrix @ matrix
            del product
        return fnp.zeros((16, 1024), dtype=fnp.float32)
