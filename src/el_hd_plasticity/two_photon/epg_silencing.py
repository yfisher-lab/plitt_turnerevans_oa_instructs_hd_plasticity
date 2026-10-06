import pathlib
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import SessionTools.two_photon as st2p
from . import session, utilities

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.gridspec import GridSpec as GS



def rho_stats(sess_df, load_row, dh_bins):
    """Compute pva magnitude binned by heading velocity.
    
    Computes mean rho (population vector average magnitude) for each bin of
    absolute heading velocity.
    
    Args:
        sess_df: Session  DataFrame.
        load_row: Callable to load preprocessed data.
        dh_bins: Bin edges for heading velocity.
        
    Returns:
        DataFrame with rho_dig: array of mean rho values for each dh bin.
    """
    rows = []
    for _, row in sess_df.iterrows():
        ts = session.GetTS(load_row(row))
        
        # Compute heading velocity and bin by absolute value
        dh = np.diff(np.unwrap(ts.heading_sm)) / ts.dt
        dh = np.concatenate([[0], dh])
        dh_dig = np.digitize(np.abs(dh), dh_bins) - 1

        # Average rho and outer ring fluorescence for each bin
        rho_dig = np.array([ts.rho[dh_dig == i].mean() for i in range(len(dh_bins))])
        F_dig = np.array([ts.outer_ring[dh_dig == i].mean() for i in range(len(dh_bins))])
        
        rows.append({
            'fly_id': row['fly_id'],
            'scan_type': row['scan_type'],
            'condition': row['condition'],
            'rho_dig': rho_dig,
            'F_dig': F_dig
        })
        
    return pd.DataFrame(rows)
