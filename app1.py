import streamlit as st
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt

def conversion_option(price, n, r, p, dt, exprice, step_dates, exercisable_date, exit_r1, exit_r2):
    """
    Calculate the value of a conversion option using a binomial tree, considering exit rates before and after vesting.

    Parameters:
    price : np.ndarray
        The binomial tree of share prices.
    n : int
        Number of steps in the tree.
    r : float
        Risk-free interest rate (continuously compounded).
    p : float
        Risk-neutral probability of an up move.
    dt : float
        Time increment per step.
    exprice : float
        Exercise (strike) price.
    step_dates : list of datetime.date
        List of dates corresponding to each step in the tree.
    exercisable_date : datetime.date
        The date from which the option can be exercised.
    exit_r1 : float
        Pre-vesting exit rate (as a decimal).
    exit_r2 : float
        Post-vesting exit rate (as a decimal).

    Returns:
    np.ndarray
        The value tree for the conversion option.
    """

    co = np.zeros_like(price)
    co[n, :] = np.maximum(price[n, :] - exprice, 0)
    for i in range(n - 1, -1, -1):
        expected_value = (p * co[i + 1, :i+1] + (1 - p) * co[i + 1, 1:i+2]) * np.exp(-r * dt)
        if step_dates[i] < exercisable_date:
            co[i, :i+1] = (1 - exit_r1 * dt) * expected_value
        else:
            forfeiture_value = np.maximum(price[i, :i+1] - exprice, 0)
            co[i, :i+1] = (1 - exit_r2 * dt) * expected_value + (exit_r2 * dt) * forfeiture_value
    return co

def conversion_option2(price, dil_price, n, r, p, dt, exprice, step_dates, exercisable_date, behav, exit_r1, exit_r2):
    """
    Calculate the value of a conversion option with dilution and early exercise behavior using a binomial tree.

    Parameters:
    price : np.ndarray
        The binomial tree of share prices.
    dil_price : np.ndarray
        The binomial tree of diluted share prices.
    n : int
        Number of steps in the tree.
    r : float
        Risk-free interest rate (continuously compounded).
    p : float
        Risk-neutral probability of an up move.
    dt : float
        Time increment per step.
    exprice : float
        Exercise (strike) price.
    step_dates : list of datetime.date
        List of dates corresponding to each step in the tree.
    exercisable_date : datetime.date
        The date from which the option can be exercised.
    behav : float
        Early exercise behavior multiplier (as a ratio).
    exit_r1 : float
        Pre-vesting exit rate (as a decimal).
    exit_r2 : float
        Post-vesting exit rate (as a decimal).

    Returns:
    np.ndarray
        The value tree for the conversion option.
    """

    co = np.zeros_like(price)
    co[n, :] = np.maximum(dil_price[n, :] - exprice, 0)
    for i in range(n - 1, -1, -1):
        expected_value = (p * co[i + 1, :i+1] + (1 - p) * co[i + 1, 1:i+2]) * np.exp(-r * dt)
        if step_dates[i] < exercisable_date:
            co[i, :i+1] = (1 - exit_r1 * dt) * expected_value
        else:
            is_early_exercise = (price[i, :i+1] > exprice * behav) & (behav > 0)
            # Must loop because exercise decision is node-specific
            for j in range(i + 1):
                if is_early_exercise[j]:
                    co[i, j] = np.maximum(dil_price[i, j] - exprice, 0)
                else:
                    forfeiture_value = np.maximum(dil_price[i, j] - exprice, 0)
                    co[i, j] = (1 - exit_r2 * dt) * (p * co[i + 1, j] + (1 - p) * co[i + 1, j + 1]) * np.exp(-r * dt) + \
                               (exit_r2 * dt) * forfeiture_value
    return co

def run_S_price(N, S, u, d, K, shares, dil, NSO):
    """
    Generate binomial trees for share prices and diluted share prices.

    Parameters:
    N : int
        Number of steps in the tree.
    S : float
        Initial share price.
    u : float
        Upward movement factor per step.
    d : float
        Downward movement factor per step.
    K : float
        Exercise (strike) price.
    shares : int
        Number of shares outstanding.
    dil : bool
        Whether to consider dilution effects (True/False).
    NSO : int
        Number of share options.

    Returns:
    tuple of np.ndarray
        Tuple containing the share price tree and the diluted price tree.
    """

    sp = np.zeros((N + 1, N + 1))
    dp = np.zeros((N + 1, N + 1))
    sp[0, 0] = S
    for i in range(1, N + 1):
        sp[i, :i] = sp[i-1, :i] * u
        sp[i, i] = sp[i-1, i-1] * d
    if not dil:
        return sp, sp
    exercisable_options = NSO
    for i in range(N + 1):
        for j in range(i + 1):
            if sp[i, j] > K:
                dp[i, j] = (sp[i, j] * shares + K * exercisable_options) / (shares + exercisable_options)
            else:
                dp[i, j] = sp[i, j]
    return sp, dp

def share_option(valuation_date, maturity_date, exercisable_date, K, S, N, behav, r,
                 sigma, q, shares, NSO, dil, exit_rate1, exit_rate2):
    """
    Use a binomial tree model to evaluate the value of a share option, considering dilution, early exercise behavior, and exit rates.

    Parameters:
    valuation_date : datetime.date
        The date on which the option is being valued.
    maturity_date : datetime.date
        The date on which the option matures.
    exercisable_date : datetime.date
        The date from which the option can be exercised (vesting date).
    K : float
        Exercise (strike) price of the option.
    S : float
        Current spot price of the underlying share.
    N : int
        Number of steps in the binomial tree.
    behav : float
        Early exercise behavior multiplier (as a ratio, e.g., 2.2 for 220%).
    r : float
        Risk-free interest rate (as a decimal, e.g., 0.05 for 5%).
    sigma : float
        Volatility of the underlying share (as a decimal, e.g., 0.3 for 30%).
    q : float
        Dividend yield (as a decimal, e.g., 0.01 for 1%).
    shares : int
        Number of shares outstanding.
    NSO : int
        Number of share options in the lot.
    dil : bool
        Whether to consider dilution effects (True/False).
    exit_rate1 : float
        Pre-vesting exit rate (as a decimal, e.g., 0.07 for 7%).
    exit_rate2 : float
        Post-vesting exit rate (as a decimal, e.g., 0.07 for 7%).

    Returns:
    float
        The calculated value per share option.
    """
    
    if not all(isinstance(d, date) for d in [valuation_date, maturity_date, exercisable_date]):
        return "Error: Invalid date provided."
    rfr = np.log(1 + r)
    life_in_days = (maturity_date - valuation_date).days
    if life_in_days <= 0 or N <= 0:
        return 0.0
    dt = (life_in_days / 365.25) / N
    u = np.exp(sigma * np.sqrt(dt))
    if u == 1: return 0.0
    d = 1 / u
    p = (np.exp((rfr - q) * dt) - d) / (u - d)
    step_dates = [valuation_date + timedelta(days=365.25 * i * dt) for i in range(N + 1)]
    sp_tree, dp_tree = run_S_price(N, S, u, d, K, shares, dil, NSO)
    if behav > 0:
        warrant_tree = conversion_option2(sp_tree, dp_tree, N, rfr, p, dt, K,
                                          step_dates, exercisable_date, behav, exit_rate1, exit_rate2)
    else:
        price_tree_to_use = dp_tree if dil else sp_tree
        warrant_tree = conversion_option(price_tree_to_use, N, rfr, p, dt, K,
                                         step_dates, exercisable_date, exit_rate1, exit_rate2)
    return warrant_tree[0, 0]

title = "SHARE OPTION VALUATION" # 标题

st.set_page_config(page_title=title, layout="wide")

# Initialize session state for storing results
if 'results' not in st.session_state:
    st.session_state.results = None
if 'calculation_mode' not in st.session_state:
    st.session_state.calculation_mode = "Single Company/Lot Valuation"

with st.sidebar:
    mode = st.selectbox(
        "Select Calculation Mode",
        ["Single Company/Lot Valuation", "Batch Calculation from Table"],
        key="calculation_mode"
    )

    if mode == "Single Company/Lot Valuation":
        with st.expander("**Global Parameters**", True):
            decimal_digits = st.number_input("Decimal Digits for Option Values", min_value=0, max_value=10, value=5, step=1)
            r = st.number_input("Risk free rate (%)", min_value=0.0, value=5.0, step=0.1) / 100
            sigma = st.number_input("Volatility (%)", min_value=0.0, value=30.0, step=1.0) / 100
            q = st.number_input("Dividend yield (%)", min_value=0.0, value=0.0, step=0.1) / 100
            shares = st.number_input("Shares outstanding", min_value=1, value=1_000_000, step=1000)
            dil = st.selectbox("Consider dilution", ["No", "Yes"], index=0) == "Yes"
        
        with st.expander("**Option Lot Parameters**", True):
            valuation_date = st.date_input("Valuation date", value=datetime(2011, 8, 30).date())
            maturity_date = st.date_input("Maturity date", value=datetime(2016, 8, 30).date())
            exercisable_date = st.date_input("Exercisable date", value=datetime(2012, 8, 30).date())
            K = st.number_input("Exercise price", min_value=0.0, value=10.0, step=0.1)
            S = st.number_input("Spot price", min_value=0.0, value=10.0, step=0.1)
            NSO = st.number_input("No. of share options", min_value=0, value=100, step=10)
            N = st.number_input("No. of steps", min_value=1, value=10, step=1)
            behav = st.number_input("Exercise behaviour multiplier (%)", min_value=0.0, value=220.0, step=0.1) / 100
            exit_rate1 = st.number_input("Pre-vesting exit rate (%)", min_value=0.0, value=0.0, step=0.1) / 100
            exit_rate2 = st.number_input("Post-vesting exit rate (%)", min_value=0.0, value=7.0, step=0.1) / 100

            params = locals().copy() # Capture all local variables as parameters

    else: # Batch Calculation from Table
        with st.expander("**Global Parameters**", True):
            st.markdown("Parameters here apply to all lots in the main table.")
            decimal_digits = st.number_input("Decimal Digits for Option Values", min_value=0, max_value=10, value=5, step=1)
            r = st.number_input("Risk free rate (%)", min_value=0.0, value=5.0, step=0.1) / 100
            sigma = st.number_input("Volatility (%)", min_value=0.0, value=30.0, step=1.0) / 100
            q = st.number_input("Dividend Yield (%)", min_value=0.0, value=0.0, step=0.1) / 100
            shares = st.number_input("Shares outstanding", min_value=1, value=1_000_000, step=1000)
            dil = st.selectbox("Consider dilution for all lots", ("No", "Yes"), index=0) == "Yes"

#st.title("📈 Share Option Valuation")
st.markdown(
f"""
<h1 style="text-align: center; border-bottom: 1px solid black; font-weight: bold; font-size: 36px; margin-bottom: 1rem;">{title}</h1>
""", unsafe_allow_html=True)

if mode == "Batch Calculation from Table":
    with st.expander("**Input Values**", True):
        dd1 = ["Decimal Digits for Option Values", "Risk free rate (%)", "Volatility (%)", "Dividend yield (%)", "Shares outstanding", 
               "Consider dilution (Yes/No)"]
        dd2 = [decimal_digits, r, sigma, q, shares, dil]

        for i, j in enumerate(dd2):
            if j==True:
                dd2[i] = "Yes"
            elif j==False:
                dd2[i] = "No"
        
        st.dataframe(pd.DataFrame({i:[j] for i, j in zip(dd1, dd2)}), hide_index=True, use_container_width=True)
        
        st.markdown(""" 
        ---
        **📄 Batch Input Table**  
        """)
        
        st.info("Edit the data for each lot below. Dates should be YYYY-MM-DD. You can add or remove lots (columns).")
        
        # Create a DataFrame with the same structure as the Excel file's tan fields
        initial_data = {
            'Lot 1': ['2011-08-30', '2016-08-30', '2011-08-30', 10.0, 10, 0, 0.0, 0.0, 10.0, 100],
            'Lot 2': ['2011-08-30', '2016-08-30', '2011-08-30', 10.0, 10, 0, 0.0, 7.0, 10.0, 100],
            'Lot 3': ['2011-08-30', '2016-08-30', '2012-08-30', 10.0, 10, 0, 0.0, 7.0, 10.0, 100],
            'Lot 4': ['2011-08-30', '2016-08-30', '2012-08-30', 10.0, 10, 220, 0.0, 7.0, 10.0, 100],
            'Lot 5': ['2011-08-30', '2016-08-30', '2011-08-30', 10.0, 10, 220, 0.0, 0.0, 10.0, 100],
        }
        index_labels = [
            "Valuation Date", "Maturity Date", "Exercisable Date", "Exercise Price",
            "No. of Steps", "Exercise Behaviour Multiplier (%)", "Pre-vesting Exit Rate (%)",
            "Post-vesting Exit Rate (%)", "Spot Price", "No. of Share Options"
        ]
        df_input = pd.DataFrame(initial_data, index=index_labels)

        edited_df = st.data_editor(
            df_input.T,
            num_rows="dynamic",
            use_container_width=True,
            key="data_editor"
        ).T
    
        calculate_button = st.button("Calculate All Lots", use_container_width=True, type="primary")
else:
    with st.expander("**Input Values**", True):
        dd1 = ["Decimal Digits for Option Values", "Risk free rate (%)", "Volatility (%)", "Dividend yield (%)", "Shares outstanding", 
               "Consider dilution (Yes/No)", "Valuation date", "Maturity date", "Exercisable date", "Exercise price", "Spot price", 
               "No. of share options", "No. of steps", "Exercise behaviour multiplier (%)", "Pre-vesting exit rate (%)", "Post-vesting exit rate (%)"]
        dd2 = [decimal_digits, r, sigma, q, shares, dil, valuation_date, maturity_date, exercisable_date, K, S, NSO, N, behav, exit_rate1, exit_rate2]
        for i, j in enumerate(dd2):
            if j==True:
                dd2[i] = "Yes"
            elif j==False:
                dd2[i] = "No"
                
        
        st.dataframe(pd.DataFrame({i:[j] for i, j in zip(dd1, dd2)}), hide_index=True, use_container_width=True)
        
        calculate_button = st.button("Calculate", use_container_width=True, type="primary")


# --- CALCULATION LOGIC ---
if calculate_button:
    if mode == "Single Company/Lot Valuation":
        try:
            # We need to pass only the arguments required by the function
            required_params = {k: v for k, v in params.items() if k in share_option.__code__.co_varnames}
            result = share_option(**required_params)
            st.session_state.results = {'value': result, 'params': params}
        except Exception as e:
            st.error(f"An error occurred during calculation: {e}")
            st.session_state.results = None

    else: # Batch Mode
        results_list = []
        has_error = False
        for lot_name in edited_df.columns:
            try:
                p = edited_df[lot_name]
                option_value = share_option(
                    valuation_date=datetime.strptime(str(p["Valuation Date"]), '%Y-%m-%d').date(),
                    maturity_date=datetime.strptime(str(p["Maturity Date"]), '%Y-%m-%d').date(),
                    exercisable_date=datetime.strptime(str(p["Exercisable Date"]), '%Y-%m-%d').date(),
                    K=float(p["Exercise Price"]),
                    S=float(p["Spot Price"]),
                    N=int(p["No. of Steps"]),
                    behav=float(p["Exercise Behaviour Multiplier (%)"]) / 100,
                    NSO=float(p["No. of Share Options"]),
                    exit_rate1=float(p["Pre-vesting Exit Rate (%)"]) / 100,
                    exit_rate2=float(p["Post-vesting Exit Rate (%)"]) / 100,
                    # Global parameters from sidebar
                    r=r, sigma=sigma, q=q, dil=dil, shares=shares
                )
                if isinstance(option_value, str):
                    st.error(f"Error in {lot_name}: {option_value}")
                    has_error = True
                else:
                    total_value = option_value * float(p["No. of Share Options"])
                    results_list.append({'Lot': lot_name, 'Per Option Value': option_value, 'Total Value': total_value})
            except Exception as e:
                st.error(f"An error occurred while processing {lot_name}: {e}")
                has_error = True

        if not has_error and results_list:
            st.session_state.results = pd.DataFrame(results_list)
        else:
            st.session_state.results = None
else:
    if mode == "Single Company/Lot Valuation":
        st.info("Please set parameters in the sidebar and click 'Calculate' to start!")
    else:
        st.info("Please set parameters in the sidebar and click 'Calculate All Lots' to start!")
            
# display results and charts
if calculate_button and (st.session_state.results is not None):
    with st.expander("**Results & Analysis**", True):
        # display for single mode
        if st.session_state.calculation_mode == "Single Company/Lot Valuation" and isinstance(st.session_state.results, dict):
            res = st.session_state.results
            p = res['params']
            digits = p.get('decimal_digits', 5)

            col1, col2 = st.columns(2)
            col1.success("**Per Option Value: "+f"{res['value']:.{digits}f}**")
            total_val = res['value'] * p['NSO']
            col2.success("**Total Value for Lot: "+f"${total_val:,.2f}**")

            sens_data = []
            base_sigma = p['sigma']
            
            # create a copy of parameters for modification
            sens_params = {k: v for k, v in p.items() if k in share_option.__code__.co_varnames}
            for v_sens in np.linspace(max(0.01, base_sigma * 0.5), base_sigma * 1.5, 20):
                sens_params['sigma'] = v_sens
                sens_data.append({
                    'Volatility': v_sens,
                    'Option Value': share_option(**sens_params)
                })
            df_sens = pd.DataFrame(sens_data)

            chart = alt.Chart(df_sens).mark_line(point=True).encode(
                x=alt.X('Volatility:Q', axis=alt.Axis(format='%')),
                y=alt.Y('Option Value:Q', axis=alt.Axis(format=f'$,.{digits}f')),
                tooltip=[alt.Tooltip('Volatility', format='.2%'), alt.Tooltip('Option Value', format=f'.{digits}f')]
            ).properties(
                title='Option Value vs. Volatility'
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)

        # display for batch mode
        elif st.session_state.calculation_mode == "Batch Calculation from Table" and isinstance(st.session_state.results, pd.DataFrame):
            df_res = st.session_state.results
            digits = st.session_state.get('decimal_digits', 5) # Retrieve from session state or default

            st.dataframe(
                df_res.style.format({
                    'Per Option Value': f'{{:.{digits}f}}',
                    'Total Value': '${:,.2f}'
                }),
                use_container_width=True
            )

            total_sum = df_res['Total Value'].apply(lambda x: round(x, 0)).sum()
            st.success("**Grand Total Value (All Lots): "+f"${total_sum:,.2f}**")

            # bar chart for comparing values
            bar_chart = alt.Chart(df_res).mark_bar().encode(
                x=alt.X('Lot:N', sort=None, title='Lot Number'),
                y=alt.Y('Per Option Value:Q', title='Per Option Value ($)'),
                color=alt.Color('Lot:N', legend=None),
                tooltip=['Lot', alt.Tooltip('Per Option Value', format=f'.{digits}f')]
            ).properties(
                title='Per Option Value Comparison by Lot'
            ).interactive()
            
            st.altair_chart(bar_chart, use_container_width=True)

            # st.markdown("---")
            # st.subheader("Benchmark Comparison (from Excel)")
            # st.markdown("`Values: Lot 1=3.51191, Lot 2=3.05534, Lot 3=3.30977, Lot 4=3.17827, Lot 5=3.3545`")
