import numpy as np
from scipy import stats
# from sklearn.ensemble import IsolationForest

def Anomaly_Detect(lst: list):
    lst = np.array(lst)
    # mean value
    mn = lst.mean()
    sigma = np.sqrt(lst.var())

    dens = stats.norm(mn, sigma)

    density = dens.pdf(lst)

    eles = density.tolist()

    return [x for x, y in zip(lst.tolist(), density) if y < np.percentile(density, 1)]