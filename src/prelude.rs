use core::ops::{Add, Div, Mul, Sub};
use rand::prelude::*;
use rand_distr::Beta;

macro_rules! constrained_f64 {
    ( $name:ident, $closure:tt, $msg:expr, $closure2:tt, $msg2:expr) => {
        #[derive(Debug, Copy, Clone)]
        pub struct $name(f64);

        impl $name {
            pub fn new(x: f64) -> Self {
                assert!(($closure)(x), $msg);
                Self(x)
            }

            pub fn new_with_variable_constraint(x: f64, y: f64) -> Self {
                assert!(($closure2)(x, y), $msg2);
                Self(x)
            }

            pub fn unwrap(self) -> f64 {
                self.0
            }

            pub fn ln(self) -> f64 {
                self.0.ln()
            }
        }

        impl Add<f64> for $name {
            type Output = f64;

            fn add(self, other: f64) -> f64 {
                self.0 + other
            }
        }

        impl Add<$name> for f64 {
            type Output = f64;

            fn add(self, other: $name) -> f64 {
                self + other.0
            }
        }

        impl Sub<f64> for $name {
            type Output = f64;

            fn sub(self, other: f64) -> f64 {
                self.0 - other
            }
        }

        impl Sub<$name> for f64 {
            type Output = f64;

            fn sub(self, other: $name) -> f64 {
                self - other.0
            }
        }

        impl Mul<f64> for $name {
            type Output = f64;

            fn mul(self, other: f64) -> f64 {
                self.0 * other
            }
        }

        impl Mul<$name> for f64 {
            type Output = f64;

            fn mul(self, other: $name) -> f64 {
                self * other.0
            }
        }

        impl Div<f64> for $name {
            type Output = f64;

            fn div(self, other: f64) -> f64 {
                self.0 / other
            }
        }

        impl Div<$name> for f64 {
            type Output = f64;

            fn div(self, other: $name) -> f64 {
                self / other.0
            }
        }
    };
}

constrained_f64!(
    Mass,
    (|x| x > 0.0),
    "Mass must be greater than zero.",
    (|x, y: f64| x > -y),
    "Mass must be greater than the negative of the discount."
);

constrained_f64!(
    Temperature,
    (|x| x >= 0.0),
    "Temperature must be greater than or equal to zero.",
    (|_x, _y| false),
    "Not supported."
);

constrained_f64!(
    Rate,
    (|x| x >= 0.0),
    "Rate must be greater than or equal to zero.",
    (|_x, _y| false),
    "Not supported."
);

impl Rate {
    pub fn resample<T: Rng>(&mut self, max: f64, shape1: f64, shape2: f64, rng: &mut T) {
        let beta = Beta::new(shape1, shape2).unwrap();
        self.0 = max * beta.sample(rng);
    }
}

constrained_f64!(
    Scale,
    (|x| x > 0.0),
    "Scale must be greater than zero.",
    (|_x, _y| false),
    "Not supported."
);

constrained_f64!(
    Reinforcement,
    (|x| (0.0..1.0).contains(&x)),
    "Reinforcement must be in [0,1).",
    (|_x, _y| false),
    "Not supported."
);

constrained_f64!(
    Discount,
    (|x| (0.0..1.0).contains(&x)),
    "Discount must be in [0,1).",
    (|_x, _y| false),
    "Not supported."
);

constrained_f64!(
    Power,
    (|x: f64| !x.is_nan()),
    "Power may not be NaN.",
    (|_x, _y| false),
    "Not supported."
);
