import numpy as np
import funsearch
from typing import Tuple, List
import pandas as pd
from scipy.optimize import minimize

def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.35) -> float:
  """Calculate Huber loss between predictions and targets"""
  error = y_true - y_pred
  is_small_error = np.abs(error) <= delta
  squared_loss = 0.5 * error ** 2
  linear_loss = delta * np.abs(error) - 0.5 * delta ** 2

  loss = is_small_error * squared_loss + (1 - is_small_error) * linear_loss
  return loss.mean()

def objective_fn(x, *args) -> float:
  data = args[0]

  sfr_true = np.array(data["log_SFR"])
  sfr_pred = np.array(priority(np.array(data["log_gas"]), np.array(data["Redshift_NED"]), *x))

  if np.isnan(sfr_pred).any() or np.isinf(sfr_pred).any() or np.any(sfr_pred == None):
    return  1e30 + 0.
  
  loss = huber_loss(np.array(sfr_true), np.array(sfr_pred))
  # check if loss is nan or not a number
  if np.isnan(loss):
    loss = 1e30 + 0.
  
  return loss


@funsearch.run
def evaluate(n: int) -> float:
  """Returns negative huber loss (to maximize) for star formation rate 
  predictions vs real star formation rate targets."""
  
  data = pd.read_csv("examples/ks_dataset_train.csv")

  objective_function = lambda x: objective_fn(x, data)

  x0 = np.random.randn(n)
  if objective_function(x0) == 1e30 + 0.:
    return -1e30 + 0.
  
  results = minimize(objective_function, x0, method="BFGS")
  loss = -results.fun

  return loss

@funsearch.evolve
def priority(log_gas_density: float, redshift: float, *constant_args: float) -> float:
  """Predicts star formation rate of galaxy from log gas density and redshift via a 
  mathematical _skeleton_ formula. A _skeleton_ formula does not contain any
  actual floating point numbers, but instead contains placeholders for them indexed
  from constant_args, which are to be optimized outside the function. For example, 
  do not generate 3.4 * log_gas_density + 2.1 * redshift, but instead generate
  constant_args[0] * log_gas_density + constant_args[1] * redshift.
  
  Args:
    log_gas_density: Input log gas density of galaxy
    redshift: Input redshift of galaxy
    *constant_args: Constants that should be independently optimized.
      
  Returns:
    Predicted star formation rate of galaxy
  """
  return 0.0