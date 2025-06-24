import streamlit as st
import datetime
from datetime import datetime as datetimenew
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.style.use("ggplot")

def binomial_tree(S, K, T, r, q, sigma, N, option_type='call', exercise_type='American', factor="No", ExerciseFactor=1.0, EffectiveDate=0.0):
    """
    Use the binomial tree model to evaluate European or American options with an adjusted exercise price.

    Parameters:
    S : Current stock price
    K : Base exercise price
    T : Time to maturity (years)
    r : Risk Free Rate (%)
    q : Dividend yield (%)
    sigma : Volatility (%)
    N : Number of steps in the tree
    option_type : 'call' or 'put'
    exercise_type : 'American' or 'European'
    factor: "Yes" or "No", when choose Yes, you can set ExerciseFactor and EffectiveDate
    ExerciseFactor : Exercise price adjustment factor (%)
    EffectiveDate : Time when the adjustment is triggered
    """
    
    dt = T / n  # Time length per step
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    p = (np.exp((np.log(1+r) - q) * dt) - d) / (u - d)  # Risk-neutral probability
    
    # Create price tree
    price_tree = np.zeros((N + 1, N + 1))
    price_tree[0, 0] = S
    
    for i in range(1, N + 1):
        price_tree[i, 0] = price_tree[i - 1, 0] * u
        for j in range(1, i + 1):
            price_tree[i, j] = price_tree[i - 1, j - 1] * d
    
    # Calculate option value tree
    option_tree = np.zeros((N + 1, N + 1))
    
    # Option value at maturity
    for j in range(N + 1):
        if option_type == 'call':
            option_tree[N, j] = max(0, price_tree[N, j] - K)
        else:
            option_tree[N, j] = max(0, K - price_tree[N, j])
    
    # Backward induction
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            # Value of holding
            hold_value = math.exp(-r * dt) * (p * option_tree[i + 1, j] + (1 - p) * option_tree[i + 1, j + 1])
            
            # Value of exercising
            if option_type == 'call':
                exercise_value = max(0, price_tree[i, j] - K)
            else:
                exercise_value = max(0, K - price_tree[i, j])
                
            if factor=="Yes":
                date = valuation_date+(maturity_date-valuation_date)/N*i
                Exercise_Price = 10000000000 if date<maturity_date else K
                Early_Exercise_Price = 0 if date<ExerciseDate else K*ExerciseFactor
                
                if price_tree[i, j] > Early_Exercise_Price:
                    option_tree[i, j] = exercise_value
                else:
                    # American options can choose to exercise immediately or continue to hold
                    option_tree[i, j] = max(hold_value, exercise_value)
            else:
                option_tree[i, j] = max(hold_value, exercise_value)
                
    return option_tree, option_tree[0, 0]

def visualize_binomial_tree(S, K, T, r, sigma, N, steps_to_show=5):
    """
    Visualize the first few steps of the binomial tree.
    
    Parameters:
    S : Current stock price
    K : Base exercise price
    T : Time to maturity (years)
    r : Risk Free Rate (%)
    q : Dividend yield (%)
    sigma : Volatility (%)
    N : Number of steps in the tree
    steps_to_show : Set steps to show
    """
    
    dt = T / n  # Time length per step
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    p = (np.exp((np.log(1+r) - q) * dt) - d) / (u - d)  # Risk-neutral probability
    
    # Create price tree
    price_tree = np.zeros((N + 1, N + 1))
    price_tree[0, 0] = S
    
    for i in range(1, steps_to_show + 1):
        price_tree[i, 0] = price_tree[i - 1, 0] * u
        for j in range(1, i + 1):
            price_tree[i, j] = price_tree[i - 1, j - 1] * d
    
    # Plot
    plt.figure(figsize=(10, 6), dpi=300)
    for i in range(steps_to_show + 1):
        for j in range(i + 1):
            if i < steps_to_show:
                plt.plot([i, i+1], [price_tree[i, j], price_tree[i+1, j+1]], color="gray", lw=0.5)
                plt.plot([i, i+1], [price_tree[i, j], price_tree[i+1, j]], color="gray", lw=0.5)
            
            plt.plot(i, price_tree[i, j], 'bo', color="gray")
            plt.text(i, price_tree[i, j], str(round(price_tree[i, j], 3)), ha="center", va="top", color="black")
    
    plt.title(f"Binomial Tree")
    plt.xlabel("Time Steps")
    plt.ylabel("Asset Price")
    plt.grid(True, lw=0.2)
    
    return plt.gcf()

title = "BINOMIAL TREE MODEL FOR THE DERIVATIVE PORTION OF CONVERTIBLE BOND"

st.set_page_config(page_title=title, layout="wide")

#st.markdown(
#f"""
#<h1 style="text-align: center; border-bottom: 1px solid black; font-weight: bold; font-size: 36px; margin-bottom: 1rem;">{title}</h1>
#""", unsafe_allow_html=True)

st.markdown("# 📈Binomial Tree Model for the Derivative Portion of Convertible Bond")
st.write("Results and analysis will appear here after calculation.")
st.markdown("## 📈Results & Analysis")

#st.markdown(
#f"""
#<h2 style="text-align: left; font-size: 25px; font-weight: bold; margin-bottom: 0.5rem;">Result & Analysis</h2>
#""", unsafe_allow_html=True)

with st.sidebar:
    option_type = st.selectbox("Option type", ["call", "put"])
    option_name = st.selectbox("The name of the option", ["American", "European"]) # Option name
    valuation_date = st.date_input("Valuation date", datetime.date(2012, 9, 30))
    maturity_date = st.date_input("Maturity date", datetime.date(2013, 3, 30)) # Maturity date
    T = (maturity_date-valuation_date).days/365.25 # In years
    #st.write(f":red[({round(T, 1)} years expected life)]")
    N = n = st.number_input('No. of steps', value=25, step=1, min_value=0) # Number of steps in the binomial tree
    #st.write(f":red[({round((maturity_date-valuation_date).days/N)} days per period on average)]")
    r = risk_free_rate = st.number_input("Risk free rate (%)", value=18.00, min_value=0.000, step=0.01)/100 # Risk-free rate
    #st.write(f":red[(continuous rate = {round(math.log(1+r)*100, 3)}%)]")
    sigma = volatility = st.number_input("Volatility (%)", value=68.07, min_value=0.00, step=0.01)/100 # Volatility
    K = price1 = st.number_input("Initial exercise price per share", value=1.00, step=0.01, min_value=0.00) # Initial exercise price per share
    S = price2 = st.number_input("Spot price per share", value=0.38, step=0.01, min_value=0.00) # Spot price per share
    q = dividend_yield = st.number_input("Dividend yield (%)", value=0.00, min_value=0.00, step=0.01)/100 # Dividend yield
    
    factor = st.selectbox("Exercise factor", ["Yes", "No"])
    if factor == "Yes":
        ExerciseFactor = exercise_factor = st.number_input("Exercise factor (%)", value=100.00, step=0.01, min_value=0.00)/100 # Exercise price adjustment factor (e.g., 1.2 means a 20% increase)
        ExerciseDate = st.date_input("Exercise factor effective date", datetime.date(2012, 9, 30)) # Time when the adjustment is triggered
        EffectiveDate = (ExerciseDate-valuation_date)/N
    else:
        ExerciseFactor = None
        ExerciseDate = None
        EffectiveDate = None
    #st.info("(assuming the holders will exercise at x% of exercise price. Empty to disable)")
    
if factor=="Yes":
    dd1 = "S, K, T, r, q, sigma, N(years), OptionType(call/put), OptionName(American/European), Factor, ExerciseFactor, EffectiveDate".split(", ")
    dd2 = [S, K, T, r, q, sigma, N, option_type, option_name, factor, ExerciseFactor, EffectiveDate]
else:
    dd1 = "S, K, T, r, q, sigma, N(years), OptionType(call/put), OptionName(American/European), Factor".split(", ")
    dd2 = [S, K, T, r, q, sigma, N, option_type, option_name, factor]

with st.expander("**Input Values**", True):
    st.write(pd.DataFrame({i:[j] for i, j in zip(dd1, dd2)}), hide_index=True, use_container_width=True)
    
    if N<=5:
        x = N
    else:
        x = 5
        
    steps = st.slider("Show steps", 0, N, value=x, step=1)
        
    button = st.button("Calculate", use_container_width=True, type="primary")

if button:
    # Calculate option price
    option_tree, price = binomial_tree(S, K, T, r, q, sigma, N, option_type, option_name, factor, ExerciseFactor, EffectiveDate)
    st.success(f"**{option_name}-style {option_type} option price: {price:.7f}**")
    with st.expander("**Binomial Tree**", True):
        fig = visualize_binomial_tree(S, K, T, r, sigma, n, steps_to_show=steps)
        st.pyplot(fig, use_container_width=True)
else:
    st.info("Please click 'Calculate' to start!")
