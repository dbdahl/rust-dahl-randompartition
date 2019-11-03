use std::ops::Add;

#[derive(Debug, Copy, Clone)]
pub struct Mass(f64);

impl Mass {
    pub fn new(x: f64) -> Self {
        assert!(x > 0.0, "Mass must be greater than zero.");
        Mass(x)
    }

    pub fn log(self) -> f64 {
        self.0.ln()
    }

    pub fn as_f64(self) -> f64 {
        self.0
    }
}

impl Add<f64> for Mass {
    type Output = f64;

    fn add(self, other: f64) -> f64 {
        let x = self.0 + other;
        assert!(x > 0.0, "Mass must be greater than zero.");
        x
    }
}

impl Add<Mass> for f64 {
    type Output = f64;

    fn add(self, other: Mass) -> f64 {
        let x = self + other.0;
        assert!(x > 0.0, "Mass must be greater than zero.");
        x
    }
}

#[derive(Debug, Copy, Clone)]
pub struct NonnegativeDouble(f64);

impl NonnegativeDouble {
    pub fn new(x: f64) -> Self {
        assert!(x >= 0.0, "Value must be greater than or equal to zero.");
        NonnegativeDouble(x)
    }

    pub fn as_f64(self) -> f64 {
        self.0
    }
}

impl Add<f64> for NonnegativeDouble {
    type Output = f64;

    fn add(self, other: f64) -> f64 {
        let x = self.0 + other;
        assert!(x >= 0.0, "Value must be greater than or equal to zero.");
        x
    }
}

impl Add<NonnegativeDouble> for f64 {
    type Output = f64;

    fn add(self, other: NonnegativeDouble) -> f64 {
        let x = self + other.0;
        assert!(x >= 0.0, "Value must be greater than or equal to zero.");
        x
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Reinforcement(f64);

impl Reinforcement {
    pub fn new(x: f64) -> Self {
        assert!(
            0.0 <= x && x < 1.0,
            format!("Reinforcement {} is not in [0,1)", x)
        );
        Reinforcement(x)
    }

    pub fn log(self) -> f64 {
        self.0.ln()
    }

    pub fn as_f64(self) -> f64 {
        self.0
    }
}

impl Add<f64> for Reinforcement {
    type Output = f64;

    fn add(self, other: f64) -> f64 {
        let x = self.0 + other;
        assert!(
            0.0 <= x && x < 1.0,
            format!("Reinforcement {} is not in [0,1)", x)
        );
        x
    }
}

impl Add<Reinforcement> for f64 {
    type Output = f64;

    fn add(self, other: Reinforcement) -> f64 {
        let x = self + other.0;
        assert!(
            0.0 <= x && x < 1.0,
            format!("Reinforcement {} is not in [0,1)", x)
        );
        x
    }
}
