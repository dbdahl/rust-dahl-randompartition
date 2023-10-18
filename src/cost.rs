use std::ops::Mul;

#[derive(Debug, Clone, Copy)]
pub struct Cost(f64);

impl Cost {
    pub fn new(x: f64) -> Option<Cost> {
        if (0.0..=2.0).contains(&x) {
            Some(Cost(x))
        } else {
            None
        }
    }
}

impl Mul<f64> for Cost {
    type Output = f64;
    fn mul(self, rhs: f64) -> f64 {
        self.0 * rhs
    }
}

impl Mul<Cost> for f64 {
    type Output = f64;
    fn mul(self, rhs: Cost) -> f64 {
        self * rhs.0
    }
}
