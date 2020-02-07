==================
Assignment1
==================
Assignment 1: Basic data analysis and simulating probability distributions
In this assignment, you will first analyse some real estate data, and then simulate some random processes corresponding to common statistical distributions and models.
Work in groups of two or three and solve the tasks described below. Write a short report containing your answers, including the plots and create a zip file containing
the report and your Python code. Alternatively, write a Jupyter notebook including your code, plots, and comments. Submit your solution through this page
(click "Submit Assignment" in the top right).

Deadline: November 13

Didactic purpose of this assignment:

practice some basic analysis of numerical data, using statistical libraries in Python,
get a gut feeling for the scenarios underlying some of the most common models used in statistics and data science;
get some experience in generating synthetic data by simulating in (simplified) models.
References
In Lecture 1, we saw how to plot histograms and compute basic descriptive statistics, and simulate some simple random processes.
Matplotlib reference documentation (Links to an external site.).
Pandas reference documentation (Links to an external site.).
NumPy random documentation (Links to an external site.).


Part 1: Real estate prices
HerePreview the document is a CSV (comma-separated values) file listing real estate sales in England between 1995 and 2016. (Actually, to make things a bit faster it's only a subset.)

1)Load the CSV file into Python. Use the Pandas function read_csv or use one of the techniques you learned in the course Introduction to Data Science.

Solution Q1::

	import pandas as pd
    import matplotlib.pyplot as plt  # basic plotting library
    import seaborn as sns  # additional plotting functions
    import scipy
    import numpy as np
    import scipy.stats as stats
    from sklearn.cluster import KMeans

    url = 'http://www.cse.chalmers.se/~richajo/dit862/houses.csv'
    data = pd.read_csv(url, header=None, usecols=[0, 1, 2, 13], names=['Id', 'price', 'date', 'location'])
    data.head()
 OUT:
    	Id	                                    price	 date	            location
    0	{02A6460F-C1F3-4977-B7ED-2A77C79302AE}	52000	2001-12-07 00:00	DERBYSHIRE
    1	{5EEB0C41-09F5-4BDF-A73E-A4E72E20BFC4}	57000	2003-09-12 00:00	WEST MIDLANDS
    2	{09975216-3588-418C-B6C5-A5704B0E77DE}	69500	2005-12-21 00:00	GREATER MANCHESTER
    3	{44CB83B9-0EC3-48D4-9B8D-B34D13D3C09E}	84000	2001-01-31 00:00	SWINDON
    4	{74828706-CBFC-406D-B7B6-394B5863E22E}	110000	1998-02-27 00:00	OXFORDSHIRE

2)The second column in the CSV file represents the price of the property. Compute basic descriptive statistics about the prices in the whole dataset: mean, median, standard deviation, minimum, and maximum.
Solution Q2::

    def header(msg):
    print('-' * 50)
    print(' [ ' + msg + ' ] ')

    def value(msg, value):
    print(msg + ':', value)


    # **************************************************************************
    # Mean Calculation:
    # **************************************************************************

    header('Mean Calculation')
    value('By DataFrame', data['price'].mean())
    value('By NumPy', np.mean(data['price']))
    value('By cipy', data['price'].mean())

    OUT::
    --------------------------------------------------
    [ Mean Calculation ]
    By DataFrame: 174386.75374
    By NumPy: 174386.75374
    By cipy: 174386.75374
    --------------------------------------------------

    # **************************************************************************
    # Median Calculation:
    # **************************************************************************

    header('Median Calculation')
    value('By DataFrame', data['price'].median())
    value('By NumPy', np.median(data['price']))
    value('By cipy', scipy.median(data['price']))

    OUT::
    --------------------------------------------------
    [ Median Calculation ]
    By DataFrame: 129000.0
    By NumPy: 129000.0
    By cipy: 129000.0

    --------------------------------------------------
    # **************************************************************************
    # Standard Deviation Calculation:
    # **************************************************************************

    header('Standard Deviation Calculation')
    value('By DataFrame', data['price'].std())
    value('By NumPy', np.std(data['price']))
    value('By cipy', scipy.std(data['price']))

    OUT::
    [ Standard Deviation Calculation ]
     By DataFrame: 351463.39776389604
     By NumPy: 351461.6404425139
     By scipy: 351461.6404425139
    --------------------------------------------------

    # **************************************************************************
    # Minimum Calculation:
    # **************************************************************************

    header('Minimum Calculation')
    value('By DataFrame', data['price'].min())
    value('By NumPy', np.min(data['price']))

    OUT::
    [ Minimum Calculation ]
    By DataFrame: 150
    By NumPy: 150
    --------------------------------------------------
    # **************************************************************************
    # Maximum Calculation:
    # **************************************************************************

    header('Maximum Calculation')
    value('By DataFrame', data['price'].max())
    value('By NumPy', np.max(data['price']))

    OUT::
    [ Maximum Calculation ]
    By DataFrame: 48465717
    By NumPy: 48465717
    --------------------------------------------------
    # **************************************************************************
    # Describe
    # **************************************************************************
    header('Describe Calculation')
    data.describe()

    OUT::
    [ Describe Calculation ]
    price
    count	1.000000e+05
    mean	1.743868e+05
    std	3.514634e+05
    min	1.500000e+02
    25%	7.400000e+04
    50%	1.290000e+05
    75%	2.070000e+05
    max	4.846572e+07


3)Plot a histogram that shows the distribution of the prices. Hint: why is it so ugly? What can you do to make it more informative?

Solution Q3::

    Ugly histogram
    Here in this histogram the price range is very high and its frequency is not uniform .
    Therefore all the frequencies are concentrated at one place due to non uniform distribution.

    plt.figure(figsize=(12, 4));
    ax1 = plt.subplot(121)
    plt.title('Normal plot of price with ugly');
    data.plot(kind='hist', ax=ax1);
    plt.xlabel('Price')



.. image:: images/DataScience/part1_q3_1.png



More Informative ::

    Plot with 90 percentile of price data.
    Since the price range is very high so we can not draw nice graph with entire data set.
    That is the reason its looks ugly.
    This graph I have taken the 90 percentile of price from minimum value of price.
    Which clearly shows that after some price the number of sale decreased exponetially.

    data.hist(alpha=1, bins='auto')
    plt.xlim(np.min(data['price']), np.percentile(data['price'], 90))
    plt.title('plot with 99 percentile');
    plt.xlabel('Price')
    plt.ylabel('Frequency')



.. image:: images/DataScience/part1_q3_2.png


4)Is real estate more expensive in London? Plot histograms for the two subsets of properties inside and outside London, respectively.
 For practical purposes, we can define "inside London" to mean that the string in the 14th column (Python indexing column 13) includes the string LONDON.


Price Inside and Outside London::

    Histogram with Price range inside and out side london
    This price range histogram clearly shows that the housing sold out side london is always higher than in london.

    inside_london = data[data.loc[:, ('location')].str.contains("LONDON")]
    price_rang_in_london = inside_london.groupby(pd.qcut(inside_london.loc[:, ('price')], 10)).size()
    value('inside london price range', price_rang_in_london.head())
    outside_london = data[~data.loc[:, ('location')].str.contains("LONDON")]
    price_rang_out_london = outside_london.groupby(pd.qcut(outside_london.loc[:, ('price')], 10)).size()
    value('outside london price range', price_rang_out_london.head())
    plt.figure(figsize=(12, 4));
    ax1 = plt.subplot(121)
    plt.title('frequency of price range inside london');
    price_rang_in_london.plot(kind='bar', ax=ax1);
    ax2 = plt.subplot(122)
    price_rang_out_london.plot(kind='bar', ax=ax2)
    plt.title('frequency of price range out side london');
    plt.xlabel('Price')
    plt.ylabel('Frequency')



.. image:: images/DataScience/part1_q4_1.png



99% Data Graph::

    Histogram with 99% of price data Inside and out side london
    This Graph shows that price in london is always higher than out side london.
    The buyer in london decreases more sharply than outside london as the price increases.

    plt.figure(figsize=(12, 3));
    ax3 = plt.subplot(121)
    inside_london.hist(alpha=1, bins='auto', ax=ax3)
    plt.xlim(np.min(inside_london.loc[:, ('price')]), np.percentile(inside_london.loc[:, ('price')], 99))
    ax3.set_xticklabels(ax3.get_xticks(), rotation=45)
    plt.title('price in london with 99 percentile');
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    ax4 = plt.subplot(122)
    outside_london.hist(alpha=1, bins='auto', ax=ax4)
    plt.xlim(np.min(outside_london.loc[:, ('price')]), np.percentile(outside_london.loc[:, ('price')], 99))
    plt.title('price out of london with 99 percentile');
    plt.xlabel('Price')
    plt.ylabel('Frequency')



.. image:: images/DataScience/part1_q4_2.png


Optional task. Make a plot that shows the average price per year.