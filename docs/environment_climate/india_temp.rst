======================
India Temperature
======================

DataSet::

    import pandas as pd
    import matplotlib.pyplot as plt  # basic plotting library
    import seaborn as sns  # additional plotting functions
    import scipy
    import scipy.stats as stats
    import numpy as np

    plt.style.use('seaborn-darkgrid')  # nicer looking plots
    url = 'https://raw.githubusercontent.com/ramashanker/dataset/master/climate/India/India_Temp_IMD_2017.csv'
    data = pd.read_csv(url)
    result=pd.concat([data.head(4), data.tail(4)])
    result
