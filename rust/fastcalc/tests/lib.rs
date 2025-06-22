#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArray2;
    use pyo3::Python;

    #[test]
    fn test_calc_position_changes() {
        Python::with_gil(|py| {
            let prev = PyArray2::<f64>::from_array(py, &[[1.0, 2.0], [3.0, 4.0]]);
            let newp = PyArray2::<f64>::from_array(py, &[[2.0, 4.0], [6.0, 8.0]]);
            let result = calc_position_changes(py, prev, newp);
            let expected = [[1.0, 2.0], [3.0, 4.0]];
            assert_eq!(result.readonly().as_array(), ndarray::arr2(&expected));
        });
    }
} 