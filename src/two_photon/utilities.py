import numpy as np 


def get_time_tick_indices(time, plot_times):
    """Find the closest indices in time array for given plot times.
    
    Args:
        time: Time array (typically pandas Series).
        plot_times: Array of time values to find indices for.
        
    Returns:
        List of indices in time array closest to each plot_time.
    """
    return [np.argmin(np.abs(time - t)) for t in plot_times]