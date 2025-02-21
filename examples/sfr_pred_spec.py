import numpy as np
import funsearch
from typing import Tuple, List

def huber_loss(y_true: float, y_pred: float, delta: float = 1.35) -> float:
  """Calculate Huber loss between predictions and targets"""
  error = y_true - y_pred
  is_small_error = np.abs(error) <= delta
  squared_loss = 0.5 * error ** 2
  linear_loss = delta * np.abs(error) - 0.5 * delta ** 2
  return squared_loss if is_small_error else linear_loss

@funsearch.run
def evaluate(data: Tuple[float, float, float]) -> float:
  """Returns negative huber loss (to maximize) for star formation rate 
  predictions vs real star formation rate targets."""
  sfr_true = data[2]
  sfr_pred = predict(data[0], data[1])
  return -huber_loss(sfr_true, sfr_pred)

@funsearch.evolve 
def predict(log_gas_density: float, redshift: float) -> float:
  """Predicts star formation rate of galaxy from log gas density and redshift
  via mathematical formula.
  
  Args:
    log_gas_density: Input log gas density of galaxy
    redshift: Input redshift of galaxy
      
  Returns:
    Predicted star formation rate of galaxy
  """
  return 0.0