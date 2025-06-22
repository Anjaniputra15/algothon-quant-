pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

use numpy::{PyArray2, IntoPyArray};
use pyo3::prelude::*;

/// Calculate position changes: new_positions - prev_positions
#[pyfunction]
fn calc_position_changes<'py>(
    py: Python<'py>,
    prev_positions: &PyArray2<f64>,
    new_positions: &PyArray2<f64>,
) -> &'py PyArray2<f64> {
    let prev = prev_positions.readonly();
    let newp = new_positions.readonly();
    let prev = prev.as_array();
    let newp = newp.as_array();
    let diff = &newp - &prev;
    diff.into_pyarray(py)
}

/// Python module definition
#[pymodule]
fn fastcalc(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calc_position_changes, m)?)?;
    Ok(())
}
