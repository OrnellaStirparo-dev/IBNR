import chainladder as cl

def run_chainladder(triangle_df):
    """Fit a Chain Ladder model and return ultimate claims."""
    triangle = cl.Triangle(triangle_df)
    model = cl.Chainladder().fit(triangle)
    ultimate = model.predict().ultimate_
    return ultimate
