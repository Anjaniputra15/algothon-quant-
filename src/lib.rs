use pyo3::prelude::*;
use ndarray::{Array1, Array2};
use statrs::distribution::{Normal, ContinuousCDF};

/// Calculate returns from price series
#[pyfunction]
fn calculate_returns_rust(prices: Vec<f64>, method: &str) -> PyResult<Vec<f64>> {
    if prices.len() < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Price series must have at least 2 elements"
        ));
    }

    let mut returns = Vec::with_capacity(prices.len() - 1);
    
    match method {
        "log" => {
            for i in 1..prices.len() {
                returns.push((prices[i] / prices[i-1]).ln());
            }
        },
        "simple" => {
            for i in 1..prices.len() {
                returns.push((prices[i] - prices[i-1]) / prices[i-1]);
            }
        },
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown method: {}", method)
            ));
        }
    }
    
    Ok(returns)
}

/// Calculate volatility from returns
#[pyfunction]
fn calculate_volatility_rust(returns: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    if returns.len() < window {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Returns series must have at least window elements"
        ));
    }

    let mut volatility = Vec::with_capacity(returns.len() - window + 1);
    
    for i in window..=returns.len() {
        let window_returns = &returns[i-window..i];
        let mean = window_returns.iter().sum::<f64>() / window_returns.len() as f64;
        let variance = window_returns.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (window_returns.len() - 1) as f64;
        volatility.push(variance.sqrt() * (252.0_f64).sqrt());
    }
    
    Ok(volatility)
}

/// Calculate Sharpe ratio
#[pyfunction]
fn calculate_sharpe_ratio_rust(returns: Vec<f64>, risk_free_rate: f64) -> PyResult<f64> {
    if returns.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Returns series cannot be empty"
        ));
    }

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter()
        .map(|&x| (x - mean_return).powi(2))
        .sum::<f64>() / (returns.len() - 1) as f64;
    let std_dev = variance.sqrt();
    
    if std_dev == 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Standard deviation cannot be zero"
        ));
    }
    
    let excess_return = mean_return - risk_free_rate / 252.0;
    Ok(excess_return / std_dev * (252.0_f64).sqrt())
}

/// Calculate maximum drawdown
#[pyfunction]
fn calculate_max_drawdown_rust(returns: Vec<f64>) -> PyResult<(f64, usize, usize)> {
    if returns.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Returns series cannot be empty"
        ));
    }

    let mut cumulative = Vec::with_capacity(returns.len() + 1);
    cumulative.push(1.0);
    
    for &ret in &returns {
        cumulative.push(cumulative.last().unwrap() * (1.0 + ret));
    }
    
    let mut max_dd = 0.0;
    let mut start_idx = 0;
    let mut end_idx = 0;
    let mut peak = cumulative[0];
    let mut peak_idx = 0;
    
    for (i, &value) in cumulative.iter().enumerate() {
        if value > peak {
            peak = value;
            peak_idx = i;
        }
        
        let drawdown = (value - peak) / peak;
        if drawdown < max_dd {
            max_dd = drawdown;
            start_idx = peak_idx;
            end_idx = i;
        }
    }
    
    Ok((max_dd, start_idx, end_idx))
}

/// Normalize data using various methods
#[pyfunction]
fn normalize_data_rust(data: Vec<f64>, method: &str) -> PyResult<Vec<f64>> {
    if data.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Data cannot be empty"
        ));
    }

    let mut normalized = Vec::with_capacity(data.len());
    
    match method {
        "zscore" => {
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            let variance = data.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / (data.len() - 1) as f64;
            let std_dev = variance.sqrt();
            
            if std_dev == 0.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Standard deviation cannot be zero for z-score normalization"
                ));
            }
            
            for &x in &data {
                normalized.push((x - mean) / std_dev);
            }
        },
        "minmax" => {
            let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            if max_val == min_val {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Max and min values cannot be equal for min-max normalization"
                ));
            }
            
            for &x in &data {
                normalized.push((x - min_val) / (max_val - min_val));
            }
        },
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown normalization method: {}", method)
            ));
        }
    }
    
    Ok(normalized)
}

/// A Python module implemented in Rust
#[pymodule]
fn algothon_quant(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_returns_rust, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_volatility_rust, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_sharpe_ratio_rust, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_max_drawdown_rust, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_data_rust, m)?)?;
    Ok(())
} 