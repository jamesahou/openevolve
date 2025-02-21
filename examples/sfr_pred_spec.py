import numpy as np
import funsearch
from typing import Tuple, List
import pandas as pd

def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.35) -> float:
  """Calculate Huber loss between predictions and targets"""
  error = y_true - y_pred
  is_small_error = np.abs(error) <= delta
  squared_loss = 0.5 * error ** 2
  linear_loss = delta * np.abs(error) - 0.5 * delta ** 2

  loss = is_small_error * squared_loss + (1 - is_small_error) * linear_loss
  return loss.mean()

@funsearch.run
def evaluate(n: int) -> float:
  """Returns negative huber loss (to maximize) for star formation rate 
  predictions vs real star formation rate targets."""
  
  data = pd.read_csv("examples/ks_dataset_train.csv")
  sfr_pred = data[['log_gas', 'Redshift_NED']].apply(lambda row: priority(*row), axis=1)
  sfr_true = data['log_SFR']

  loss = -huber_loss(np.array(sfr_true), np.array(sfr_pred))
  if np.isnan(loss):
    loss = -1e30 + 0.

  return loss

@funsearch.evolve
def priority(log_gas_density: float, redshift: float) -> float:
  """Predicts star formation rate of galaxy from log gas density and redshift
  via mathematical formula.
  
  Args:
    log_gas_density: Input log gas density of galaxy
    redshift: Input redshift of galaxy
      
  Returns:
    Predicted star formation rate of galaxy
  """
  return 0.0