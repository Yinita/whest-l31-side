"""Harness fault fixture only; external supervision must kill this process."""
import signal
import time

from whestbench import BaseEstimator


class Estimator(BaseEstimator):
    def predict(self, mlp, budget):
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
        while True:
            time.sleep(1)
