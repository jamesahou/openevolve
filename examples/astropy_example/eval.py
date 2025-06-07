import numpy as np
from astropy.coordinates import (
    SkyCoord,
    FK5,
    Latitude,
    Angle,
    ICRS,
    concatenate,
    UnitSphericalRepresentation,
    CartesianRepresentation,
    match_coordinates_sky,
)
from astropy import units as u
from astropy.time import Time
import time
import funsearch

@funsearch.run
def evaluate_concatenate():
    start_time = time.time()
    icrs_array = ICRS(
            ra=np.random.random(10000) * u.deg, dec=np.random.random(10000) * u.deg
    )
    concatenate((icrs_array, icrs_array))
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    result = evaluate_concatenate()
    print(f"Time taken to concatenate ICRS arrays: {result:.4f} seconds")