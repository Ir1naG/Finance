import numpy as np

def discounted_cash_flow_valuation(free_cash_flows, discount_rate, long_term_growth_rate):
    """
    Estimate the value of a company using a simplified Discounted Cash Flow (DCF).
    
    Parameters
    ----------
    free_cash_flows : list or np.array of float
        Projected free cash flows for each forecast year (in currency units).
    discount_rate : float
        Discount rate (e.g., WACC) in decimal form (e.g., 0.10 for 10%).
    long_term_growth_rate : float
        Annual growth rate of FCF beyond the final forecast year in decimal form.

    Returns
    -------
    float
        Estimated enterprise value (in the same currency units as free_cash_flows).
    """
    # Number of forecasted years
    n = len(free_cash_flows)
    
    # Calculate discount factors (one for each year)
    discount_factors = [(1 + discount_rate)**year for year in range(1, n + 1)]
    
    # Present value of each year's forecasted free cash flow
    discounted_fcfs = [
        fcf / df for fcf, df in zip(free_cash_flows, discount_factors)
    ]
    
    # Calculate the terminal value using the perpetual growth model
    # TV_n = (FCF_n+1) / (r - g) = (FCF_n * (1+g)) / (r - g)
    terminal_value = (
        free_cash_flows[-1] * (1 + long_term_growth_rate) 
        / (discount_rate - long_term_growth_rate)
    )
    
    # Discount terminal value back to present
    discounted_terminal_value = terminal_value / ((1 + discount_rate) ** n)
    
    # Sum the discounted free cash flows + discounted terminal value
    enterprise_value = sum(discounted_fcfs) + discounted_terminal_value
    
    return enterprise_value

if __name__ == "__main__":
    # Example data for NBIS:
    # 1. Projected FCFs (in millions)
    nbis_fcfs = [5.0, 7.0, 9.0, 12.0, 15.0]
    
    # 2. Discount rate (WACC), e.g. 10%
    wacc = 0.10
    
    # 3. Long-term growth rate, e.g. 3%
    long_term_growth = 0.03
    
    # 4. Calculate the enterprise value using our DCF function
    nbis_enterprise_value = discounted_cash_flow_valuation(
        free_cash_flows=nbis_fcfs,
        discount_rate=wacc,
        long_term_growth_rate=long_term_growth
    )
    
    # 5. Assume NBIS has $10M in total debt and $2M in cash => net debt of $8M
    net_debt = 10.0 - 2.0  # = 8M
    
    # 6. Calculate equity value (EV - Net Debt)
    nbis_equity_value = nbis_enterprise_value - net_debt
    
    # Print results
    print(f"NBIS Enterprise Value (DCF): ${nbis_enterprise_value:,.2f} million")
    print(f"NBIS Equity Value:           ${nbis_equity_value:,.2f} million")
