module AlgothonQuant

using DataFrames
using CSV
using JSON3
using Statistics
using LinearAlgebra
using Optim
using JuMP
using Ipopt
using Plots
using StatsBase
using Distributions
using TimeSeries
using Random
using Dates
using Printf
using Logging

export calculate_returns_julia,
       calculate_volatility_julia,
       calculate_sharpe_ratio_julia,
       calculate_max_drawdown_julia,
       normalize_data_julia

"""
Calculate returns from price series
"""
function calculate_returns_julia(prices::Vector{Float64}; method::String="log")
    if length(prices) < 2
        error("Price series must have at least 2 elements")
    end
    
    if method == "log"
        return [log(prices[i] / prices[i-1]) for i in 2:length(prices)]
    elseif method == "simple"
        return [(prices[i] - prices[i-1]) / prices[i-1] for i in 2:length(prices)]
    else
        error("Unknown method: $method")
    end
end

"""
Calculate volatility from returns
"""
function calculate_volatility_julia(returns::Vector{Float64}; window::Int=252)
    if length(returns) < window
        error("Returns series must have at least $window elements")
    end
    
    volatility = Float64[]
    for i in window:length(returns)
        window_returns = returns[i-window+1:i]
        vol = std(window_returns) * sqrt(252)
        push!(volatility, vol)
    end
    
    return volatility
end

"""
Calculate Sharpe ratio
"""
function calculate_sharpe_ratio_julia(returns::Vector{Float64}; risk_free_rate::Float64=0.02)
    if isempty(returns)
        error("Returns series cannot be empty")
    end
    
    mean_return = mean(returns)
    std_return = std(returns)
    
    if std_return == 0.0
        error("Standard deviation cannot be zero")
    end
    
    excess_return = mean_return - risk_free_rate / 252
    return excess_return / std_return * sqrt(252)
end

"""
Calculate maximum drawdown
"""
function calculate_max_drawdown_julia(returns::Vector{Float64})
    if isempty(returns)
        error("Returns series cannot be empty")
    end
    
    cumulative = [1.0]
    for ret in returns
        push!(cumulative, cumulative[end] * (1 + ret))
    end
    
    max_dd = 0.0
    start_idx = 1
    end_idx = 1
    peak = cumulative[1]
    peak_idx = 1
    
    for (i, value) in enumerate(cumulative)
        if value > peak
            peak = value
            peak_idx = i
        end
        
        drawdown = (value - peak) / peak
        if drawdown < max_dd
            max_dd = drawdown
            start_idx = peak_idx
            end_idx = i
        end
    end
    
    return max_dd, start_idx, end_idx
end

"""
Normalize data using various methods
"""
function normalize_data_julia(data::Vector{Float64}; method::String="zscore")
    if isempty(data)
        error("Data cannot be empty")
    end
    
    if method == "zscore"
        mean_val = mean(data)
        std_val = std(data)
        
        if std_val == 0.0
            error("Standard deviation cannot be zero for z-score normalization")
        end
        
        return [(x - mean_val) / std_val for x in data]
        
    elseif method == "minmax"
        min_val = minimum(data)
        max_val = maximum(data)
        
        if max_val == min_val
            error("Max and min values cannot be equal for min-max normalization")
        end
        
        return [(x - min_val) / (max_val - min_val) for x in data]
        
    elseif method == "robust"
        median_val = median(data)
        mad = median([abs(x - median_val) for x in data])
        
        if mad == 0.0
            error("Median absolute deviation cannot be zero for robust normalization")
        end
        
        return [(x - median_val) / mad for x in data]
        
    else
        error("Unknown normalization method: $method")
    end
end

"""
Optimize portfolio weights using mean-variance optimization
"""
function optimize_portfolio_julia(returns::Matrix{Float64}; 
                                risk_free_rate::Float64=0.02,
                                target_return::Union{Float64, Nothing}=nothing)
    n_assets = size(returns, 2)
    
    # Calculate mean returns and covariance matrix
    mean_returns = vec(mean(returns, dims=1))
    cov_matrix = cov(returns)
    
    # Create optimization model
    model = Model(Ipopt.Optimizer)
    
    # Variables: portfolio weights
    @variable(model, 0 <= w[1:n_assets] <= 1)
    
    # Constraint: weights sum to 1
    @constraint(model, sum(w) == 1.0)
    
    # Optional constraint: target return
    if !isnothing(target_return)
        @constraint(model, sum(w[i] * mean_returns[i] for i in 1:n_assets) >= target_return)
    end
    
    # Objective: minimize portfolio variance
    @objective(model, Min, sum(w[i] * w[j] * cov_matrix[i,j] for i in 1:n_assets, j in 1:n_assets))
    
    # Solve
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return value.(w)
    else
        error("Optimization failed")
    end
end

"""
Calculate Value at Risk (VaR)
"""
function calculate_var_julia(returns::Vector{Float64}; confidence_level::Float64=0.95)
    if isempty(returns)
        error("Returns series cannot be empty")
    end
    
    if confidence_level <= 0.0 || confidence_level >= 1.0
        error("Confidence level must be between 0 and 1")
    end
    
    return quantile(returns, 1.0 - confidence_level)
end

"""
Calculate Conditional Value at Risk (CVaR)
"""
function calculate_cvar_julia(returns::Vector{Float64}; confidence_level::Float64=0.95)
    if isempty(returns)
        error("Returns series cannot be empty")
    end
    
    if confidence_level <= 0.0 || confidence_level >= 1.0
        error("Confidence level must be between 0 and 1")
    end
    
    var = calculate_var_julia(returns, confidence_level=confidence_level)
    tail_returns = returns[returns .<= var]
    
    if isempty(tail_returns)
        return var
    else
        return mean(tail_returns)
    end
end

end # module 