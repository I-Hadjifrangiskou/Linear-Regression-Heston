# Linear-Regression-Heston

This project uses a linear regression algorithm implemented on Python, and monte carlo simulations to optimise the parameters of the Heston model. These are used to check how accurately the algorithm predicts stock prices as a function of strike prices using this as the underlying model. The Heston model is solved by following the Quadratic-Exponential (QE) algorithm as illustrated by Leif Andersen in his 2008 paper: Efficient Simulation of the Heston Stochastic Volatility Model. Initial market conditions are taken to be the ones 30 days in the past, and today's market prices are the ones on which parameters are optimised. Further details are found in the comments of the Python script heston_model_linear_reg.py.

The repository also includes a plot of the predicted prices and the real prices from a past run, labelled lin_reg.png.      
