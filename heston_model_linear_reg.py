import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
import scipy.stats

startTime = datetime.now()

# Define how long ago (in days) you would like to predict from
period_days = 30

# Use real-world market option prices from Yahoo Finance
ticker = "SPY" # Using SPY as a proxy for European options
spy = yf.Ticker(ticker)
expirations = spy.options  # Get available expiry dates

# Select a near-term expiration
opt_data = spy.option_chain(expirations[1])  
calls = opt_data.calls[['strike', 'lastPrice', 'impliedVolatility']]

# Remove NaN values
calls = calls.dropna()
calls = calls[calls['impliedVolatility'] > 0]

# Fetch real strikes for visualization
strikes = calls['strike'].values
implied_vol = calls['impliedVolatility'].values

# Fetch risk-free rate from Yahoo Finance (3-month T-bill rate) and market prices for visualisation
t_bill = yf.Ticker("^IRX")
r_real = t_bill.history(period="1d")['Close'].iloc[-1] / 100  # Convert to decimal
market_prices = calls['lastPrice'].values

# Features for linear regression: Strike price, Implied Volatility, Risk-free rate
X = calls[['strike', 'impliedVolatility']]
X['risk_free_rate'] = r_real

# Targets (Heston parameters) - Replace with real calibration data if available
y = pd.DataFrame({
    'theta': np.random.uniform(0.01, 0.1, len(X)),
    'kappa': np.random.uniform( 0.5, 2,   len(X)),
    'sigma': np.random.uniform( 0.1, 1.0, len(X)),
    'rho':   np.random.uniform(-0.5, 0.5, len(X))
})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model 
model = LinearRegression()
model.fit(X_train, y_train)

# Predict Heston parameters for all market strikes
y_pred = model.predict(X)

# Extract predicted parameters
pred_theta = y_pred[:, 0]
pred_kappa = y_pred[:, 1]
pred_sigma = y_pred[:, 2]
pred_rho = y_pred[:, 3]


# This algorithm follows the Quadratic-Exponential (QE) implementation by Leif Andersen (Efficient Simulation of the Heston Stochastic Volatility Model)
def heston_model_montecarlo(T, dt, monte_runs, S0, K, mu, kappa, theta, epsilon, rho, v0):
  

    # Constants for Quadratic-Exponential scheme (Andersen 2008)
    psi_c  = 1.5  # Arbitrary value 1 < psi_c < 2 used for switching rule. According to Andersen, this makes little difference
    gamma1 = 1.0  # Constants gamma1 and gamma2 give the proposed discretization scheme (gamma1 = 1, gamma2 = 0 is Euler for example)
    gamma2 = 0.0

    # The following are timestep dependent parameters that are cached before the loop as they are independent of the dynamical variables
    k1 = np.exp(-kappa * dt)
    k2 = epsilon**2 * k1 / kappa * (1 - k1)
    k3 = theta * epsilon**2 / (2 * kappa) * (1 - k1)**2
    K0 = -rho * kappa * theta * dt / epsilon
    K1 = gamma1 * dt * (kappa * rho / epsilon - 0.5) - rho / epsilon
    K2 = gamma2 * dt * (kappa * rho / epsilon - 0.5) + rho / epsilon
    K3 = gamma1 * dt * (1 - rho**2)
    K4 = gamma2 * dt * (1 - rho**2)
    
    # Initialize arrays for log stock prices and volatilities
    n_steps = int(T / dt)
    lnS     = np.zeros((monte_runs, n_steps))
    v       = np.zeros((monte_runs, n_steps))

    # Set initial conditions for stock price (S) and volatility (v)
    lnS[:, 0] = np.log(S0)
    v[:, 0]   = v0

    # Generate random variables for Monte Carlo paths
    Uv = np.random.uniform(0, 1, (monte_runs, n_steps))
    Zx = np.random.normal(0, 1, (monte_runs, n_steps))

    # Update loop for ln(S) and v
    for t in range(1, n_steps):

        # Parameters m, s^2 and psi as defined by Andersen
        m = theta + (v[:, t - 1] - theta) * k1
        s2 = v[:, t - 1] * k2 + k3
        psi = np.maximum(s2 / m**2, 1e-5)  # Prevent division by zero

        # Volatility update
        for i in range(monte_runs):

            # Case when psi < psi_c, the critical value defined above
            if psi[i] < psi_c:

                # Calculating parameters a, b as defined by Andersen and the standard normal variable Z_v 
                b2 = 2 / psi[i] - 1 + np.sqrt(2 / psi[i]) * np.sqrt(np.maximum(2 / psi[i] - 1, 0))
                a = m[i] / (1 + b2)
                Zv = scipy.stats.norm.ppf(Uv[i, t])

                # Updating v
                v[i, t] = np.maximum(a * (np.sqrt(b2) + Zv) ** 2, 0)

            # Case when psi > psi_c
            else:

                # Calculating parameters p and beta as defined by Andersen
                p = (psi[i] - 1) / (psi[i] + 1)
                beta = 2 / (m[i] * (psi[i] + 1))

                # Piecewise function PSI inverse as defined by Andersen
                if Uv[i, t] < p:
                    v[i, t] = 0

                else:
                    v[i, t] = np.maximum(np.log((1 - p) / np.maximum((1 - Uv[i, t]), 1e-5)) / beta, 0)

        # Updating ln(S)
        lnS[:, t] = lnS[:, t - 1] + K0 + mu * dt + K1 * v[:, t - 1] + K2 * v[:, t] + np.sqrt(K3 * v[:, t - 1] + K4 * v[:, t]) * Zx[:, t]

    # Exponentiating to get stock price (S) as a functon of time
    S = np.exp(lnS)

    # Payoff for a European call option
    payoff = np.maximum(S[:, -1] - K, 0)

    # Estimate the option price under risk-neutral measure
    price = np.exp(-mu * T) * np.mean(payoff)
    
    return price

# Predict Option Prices Using Monte Carlo 
S0_real = spy.history(period = f"{period_days}d")['Close'].iloc[-1]  # Current stock price

# Time to maturity and timestep size
expiry_date = datetime.strptime(expirations[1], "%Y-%m-%d")
today = datetime.today()
#T_real = (expiry_date - today).days / 365
T_real = period_days / 365
dt = T_real / 100
monte_runs = 50  # Number of Monte Carlo paths


# Compute model-predicted option prices
predicted_prices = np.array([
    heston_model_montecarlo(T_real, dt, monte_runs, S0_real, K, r_real,
                            kappa, theta, sigma, rho, v0)
    for K, v0, theta, kappa, sigma, rho in zip(strikes, implied_vol , pred_theta, pred_kappa, pred_sigma, pred_rho)
])

# Plot and save Linear Regression predictions vs. Real Market Prices
plt.figure(figsize = (10, 5))
plt.plot(strikes, market_prices, 'o-', label = "Real Market Prices", color = 'blue')
plt.plot(strikes, predicted_prices, 's-', label = "Linear Regression Heston Predictions", color = 'red')
plt.xlabel("Strike Price")
plt.ylabel("Option Price")
plt.legend()
plt.title("Model-Predicted vs. Real Market Option Prices (Linear Regression)")
plt.savefig("lin_reg.png")
plt.show()
