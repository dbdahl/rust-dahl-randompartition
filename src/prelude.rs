use core::ops::{Add, Mul, Sub};

macro_rules! constrained_f64 {
    ( $name:ident, $closure:tt, $msg:expr ) => {
        #[derive(Debug, Copy, Clone)]
        pub struct $name(f64);

        impl $name {
            pub fn new(x: f64) -> Self {
                assert!($closure(x), $msg);
                Self(x)
            }

            pub fn unwrap(self) -> f64 {
                self.0
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
    };
}

constrained_f64!(Mass, (|x| x > 0.0), "Mass must be greater than zero.");

constrained_f64!(
    Temperature,
    (|x| x >= 0.0),
    "Temperature must be greater than or equal to zero."
);

constrained_f64!(
    UinNGGP,
    (|x| x >= 0.0),
    "Temperature must be greater than or equal to zero."
);

constrained_f64!(
    Rate,
    (|x| x >= 0.0),
    "Rate must be greater than or equal to zero."
);

constrained_f64!(
    Reinforcement,
    (|x| 0.0 <= x && x < 1.0),
    "Reinforcement must be in [0,1)."
);

constrained_f64!(
    Discount,
    (|x| 0.0 <= x && x < 1.0),
    "Discount must be in [0,1)."
);
