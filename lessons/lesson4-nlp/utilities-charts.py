import numpy as np, matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from datasets import Dataset,DatasetDict
from numpy.random import normal,seed,uniform
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

## VISUALIZE UTILITIES
def add_noise(x, mult, add): return x * (1+noise(x,mult)) + noise(x,add)

def plot_ds():

    def f(x): return -3*x**2 + 2*x + 20
    def plot_function(f, min=-2.1, max=2.1, color='r'):
        x = np.linspace(min,max, 100)[:,None]
        plt.plot(x, f(x), color)
        def noise(x, scale): return normal(scale=scale, size=x.shape)
    def visualizeData():
        np.random.seed(42)
        x = np.linspace(-2, 2, num=20)[:,None]
        y = add_noise(f(x), 0.2, 1.3)
        plt.scatter(x,y)

        plot_function(f, color='b')


    def plot_poly(degree):
        x = np.linspace(-2, 2, num=20)[:,None]
        y = add_noise(f(x), 0.2, 1.3)
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(x, y)
        plt.scatter(x,y)
        plot_function(model.predict)
    
    visualizeData()
    plot_poly(10)
    plt.show()

def show_correlation():
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing(as_frame=True)
    housing = housing['data'].join(housing['target']).sample(1000, random_state=52)
    
    print(housing.head())

    # Show all coefficient correlations for all combinations of columns
    np.set_printoptions(precision=2, suppress=True)
    np.corrcoef(housing, rowvar=False)

    np.corrcoef(housing.MedInc, housing.MedHouseVal)
    # return single coefficient to make splotting easier

    corr(housing.MedInc, housing.MedHouseVal)

    def show_corr(df, a, b):
        x,y = df[a],df[b]
        plt.scatter(x,y, alpha=0.5, s=4)
        plt.title(f'{a} vs {b}; r: {corr(x, y):.2f}')

    subset = housing[housing.AveRooms<15]
    #show_corr(subset, 'MedInc', 'AveRooms')
    #show_corr(housing, 'MedInc', 'MedHouseVal')
    ##show_corr(housing, 'Population', 'AveRooms')
    show_corr(subset, 'HouseAge', 'AveRooms')
    plt.show()
