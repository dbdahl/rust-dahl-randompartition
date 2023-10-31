use core::ops::{Add, Div, Mul, Neg, Sub};

#[allow(clippy::redundant_closure_call)]
macro_rules! constrained_f64 {
    ( $name:ident, $x:ident, $closure:expr ) => {
        #[derive(Debug, Copy, Clone)]
        pub struct $name(f64);

        impl $name {
            pub fn new($x: f64) -> Option<Self> {
                if !$x.is_finite() {
                    None
                } else if $closure {
                    Some(Self($x))
                } else {
                    None
                }
            }

            pub fn set(&mut self, $x: f64) -> Option<()> {
                if !$x.is_finite() {
                    None
                } else if $closure {
                    self.0 = $x;
                    Some(())
                } else {
                    None
                }
            }

            pub fn get(&self) -> f64 {
                self.0
            }

            pub fn ln(&self) -> f64 {
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

        impl Neg for $name {
            type Output = f64;

            fn neg(self) -> f64 {
                -self.0
            }
        }
    };
}

constrained_f64!(Concentration, x, x > 0.0);
constrained_f64!(Discount, x, (0.0..1.0).contains(&x));
constrained_f64!(Shape, x, x > 0.0);
constrained_f64!(Rate, x, x > 0.0);
constrained_f64!(Scale, x, x > 0.0);
constrained_f64!(ScalarShrinkage, x, x > 0.0);
constrained_f64!(Temperature, x, x >= 0.0);
constrained_f64!(Cost, x, (0.0..=2.0).contains(&x));

impl Concentration {
    pub fn new_with_discount(x: f64, discount: Discount) -> Option<Self> {
        if x > -discount {
            Some(Self(x))
        } else {
            None
        }
    }
}

impl Scale {
    pub fn to_rate(self) -> Rate {
        Rate(1.0 / self.0)
    }
    pub fn to_shrinkage(self) -> ScalarShrinkage {
        ScalarShrinkage(1.0 / self.0)
    }
}

impl Rate {
    pub fn to_scale(self) -> Scale {
        Scale(1.0 / self.0)
    }
}

impl Discount {
    pub fn zero() -> Self {
        Self(0.0)
    }
}

impl Cost {
    pub fn one() -> Self {
        Self(1.0)
    }
}
