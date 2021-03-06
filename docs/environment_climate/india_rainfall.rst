======================
Rain_Fall_India
======================
#Data authenticity::

    https://data.gov.in/catalog/rainfall-india?filters%5Bfield_catalog_reference%5D=1090541&format=json&offset=
    0&limit=6&sort%5Bcreated%5D=desc

Rain fall Data Set ::

    import pandas as pd
    import matplotlib.pyplot as plt  # basic plotting library
    import seaborn as sns  # additional plotting functions
    import scipy
    import scipy.stats as stats
    import numpy as np

    plt.style.use('seaborn-darkgrid')  # nicer looking plots
    url = 'https://raw.githubusercontent.com/ramashanker/dataset/master/climate/India/India_Rain_Fall.csv'
    data = pd.read_csv(url)
    result=pd.concat([data.head(4), data.tail(4)])
    result

.. image:: ../images/environment_climate/india_rain_fall/dataset.png

Scatter Plot Data::

    def scatter_plot(x,y):
        plt.figure(figsize=(32, 4));
        ax1 = plt.subplot(121)
        plt.scatter(x, y)
        plt.show()
    scatter_plot(data['YEAR'],data['ANNUAL'])

.. image:: ../images/environment_climate/india_rain_fall/scatterplot.png



Plotting linear and cluster to verify Annual rain fall::

    #Plotting yearly rain fall:
    x=data['YEAR']
    y=data['ANNUAL']
    w0,w1=calculate_linear_equation(x,y)
    plot_data(w0,w1,x,y)
    plt.figure(figsize=(15,6))
    sns.kdeplot(x, y, shade=True);

.. image:: ../images/environment_climate/india_rain_fall/annual.png


Rain Fall June::

    x=data['YEAR']
    y=data['JUN']
    w0,w1=calculate_linear_equation(x,y)
    plot_data(w0,w1,x,y)
    plt.figure(figsize=(15,6))
    sns.kdeplot(x, y, shade=True);

.. image:: ../images/environment_climate/india_rain_fall/june.png


Rain Fall July::

    x=data['YEAR']
    y=data['JUL']
    w0,w1=calculate_linear_equation(x,y)
    plot_data(w0,w1,x,y)
    plt.figure(figsize=(15,6))
    sns.kdeplot(x, y, shade=True);

.. image:: ../images/environment_climate/india_rain_fall/july.png


Rain Fall August::

    x=data['YEAR']
    y=data['AUG']
    w0,w1=calculate_linear_equation(x,y)
    plot_data(w0,w1,x,y)
    plt.figure(figsize=(15,6))
    sns.kdeplot(x, y, shade=True);

.. image:: ../images/environment_climate/india_rain_fall/august.png


Rain Fall September::

    x=data['YEAR']
    y=data['SEP']
    w0,w1=calculate_linear_equation(x,y)
    plot_data(w0,w1,x,y)
    plt.figure(figsize=(15,6))
    sns.kdeplot(x, y, shade=True);

.. image:: ../images/environment_climate/india_rain_fall/september.png


Rain Fall October::

    x=data['YEAR']
    y=data['OCT']
    w0,w1=calculate_linear_equation(x,y)
    plot_data(w0,w1,x,y)
    plt.figure(figsize=(15,6))
    sns.kdeplot(x, y, shade=True);

.. image:: ../images/environment_climate/india_rain_fall/october.png


Rain Fall November::

    x=data['YEAR']
    y=data['NOV']
    w0,w1=calculate_linear_equation(x,y)
    plot_data(w0,w1,x,y)
    plt.figure(figsize=(15,6))
    sns.kdeplot(x, y, shade=True);

.. image:: ../images/environment_climate/india_rain_fall/november.png


Rain Fall December::

    x=data['YEAR']
    y=data['DEC']
    w0,w1=calculate_linear_equation(x,y)
    plot_data(w0,w1,x,y)
    plt.figure(figsize=(15,6))
    sns.kdeplot(x, y, shade=True);

.. image:: ../images/environment_climate/india_rain_fall/december.png


Rain Fall January::

    x=data['YEAR']
    y=data['JAN']
    w0,w1=calculate_linear_equation(x,y)
    plot_data(w0,w1,x,y)
    plt.figure(figsize=(15,6))
    sns.kdeplot(x, y, shade=True);

.. image:: ../images/environment_climate/india_rain_fall/january.png



Rain Fall February::

    x=data['YEAR']
    y=data['FEB']
    w0,w1=calculate_linear_equation(x,y)
    plot_data(w0,w1,x,y)
    plt.figure(figsize=(15,6))
    sns.kdeplot(x, y, shade=True);

.. image:: ../images/environment_climate/india_rain_fall/february.png


Rain Fall March::

    x=data['YEAR']
    y=data['MAR']
    w0,w1=calculate_linear_equation(x,y)
    plot_data(w0,w1,x,y)
    plt.figure(figsize=(15,6))
    sns.kdeplot(x, y, shade=True);

.. image:: ../images/environment_climate/india_rain_fall/march.png


Rain Fall April::

    x=data['YEAR']
    y=data['APR']
    w0,w1=calculate_linear_equation(x,y)
    plot_data(w0,w1,x,y)
    plt.figure(figsize=(15,6))
    sns.kdeplot(x, y, shade=True);

.. image:: ../images/environment_climate/india_rain_fall/april.png