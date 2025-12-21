import chainladder as cl
import numpy as np
import pandas as pd

def run_bornhuetter_ferguson(triangle_df, apriori=None, apriori_sigma=None):
    """
    Fit a Bornhuetter-Ferguson model and return ultimate claims.
    
    The Bornhuetter-Ferguson method combines Chain Ladder development factors
    with prior estimates of ultimate claims.
    
    Args:
        triangle_df: Claims triangle data (can be DataFrame or chainladder Triangle)
        apriori: Prior estimate of ultimate claims (optional, will use Chain Ladder if not provided)
        apriori_sigma: Prior estimate of standard deviation (optional)
    
    Returns:
        Ultimate claims estimate
    """
    # Convert to chainladder Triangle if needed
    if isinstance(triangle_df, pd.DataFrame):
        triangle = cl.Triangle(triangle_df)
    else:
        triangle = triangle_df
    
    # If apriori not provided, use Chain Ladder as prior
    if apriori is None:
        cl_model = cl.Chainladder().fit(triangle)
        apriori = cl_model.predict().ultimate_
    
    # Fit Bornhuetter-Ferguson model
    bf_model = cl.BornhuetterFerguson(apriori=apriori, apriori_sigma=apriori_sigma)
    bf_model = bf_model.fit(triangle)
    ultimate = bf_model.predict().ultimate_
    
    return ultimate
