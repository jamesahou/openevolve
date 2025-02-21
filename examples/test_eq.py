import numpy as np
import pickle

def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float=1.35) -> float:
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * np.abs(error) - 0.5 * delta ** 2
    return squared_loss if is_small_error else linear_loss

def eq(log_gas_density: float, redshift: float) -> float:
    out =  10**(0.3 * (log_gas_density - 12) + 0.8 * (redshift - 1)) 
    return out

data = pickle.load(open("/home/jamesh/LASR/funsearch/examples/ks_dataset_test.pickle", "rb"))
mean_loss = 0
for (log_gas_density, redshift, log_sfr) in data:
    log_sfr_pred = eq(log_gas_density, redshift)
    loss = huber_loss(log_sfr, log_sfr_pred)
    print(log_sfr_pred, log_sfr)
    print(loss)
    mean_loss += loss

print(mean_loss / len(data))
