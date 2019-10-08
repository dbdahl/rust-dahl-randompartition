use std::ops::Add;

#[derive(Debug, Copy, Clone)]
pub struct Mass(f64);

impl Mass {
    pub fn new(x: f64) -> Self {
        assert!(x > 0.0, "Mass must be greater than zero.");
        Mass(x)
    }

    pub fn as_f64(self) -> f64 {
        self.0
    }
}

impl Add<f64> for Mass {
    type Output = f64;

    fn add(self, other: f64) -> f64 {
        let ans = self.0 + other;
        assert!(ans > 0.0);
        ans
    }
}

impl Add<Mass> for f64 {
    type Output = f64;

    fn add(self, other: Mass) -> f64 {
        let ans = self + other.0;
        assert!(ans > 0.0);
        ans
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Rate(f64);

impl Rate {
    pub fn new(x: f64) -> Self {
        assert!(x > 0.0, "Rate must be greater than zero.");
        Rate(x)
    }

    pub fn as_f64(self) -> f64 {
        self.0
    }
}

impl Add<f64> for Rate {
    type Output = f64;

    fn add(self, other: f64) -> f64 {
        let ans = self.0 + other;
        assert!(ans > 0.0);
        ans
    }
}

impl Add<Rate> for f64 {
    type Output = f64;

    fn add(self, other: Rate) -> f64 {
        let ans = self + other.0;
        assert!(ans > 0.0);
        ans
    }
}
