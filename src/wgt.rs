use crate::prelude::Rate;

#[derive(Debug, Clone)]
pub struct Weights(Vec<f64>);

impl Weights {
    pub fn zero(n_items: usize) -> Weights {
        Weights(vec![0.0; n_items])
    }

    pub fn from_rate(rate: Rate, n_items: usize) -> Weights {
        Weights(vec![rate.unwrap(); n_items])
    }

    pub fn constant(value: f64, n_items: usize) -> Option<Weights> {
        if value.is_nan() || value.is_infinite() || value < 0.0 {
            return None;
        }
        Some(Weights(vec![value; n_items]))
    }

    pub fn from(w: &[f64]) -> Option<Weights> {
        for ww in w.iter() {
            if ww.is_nan() || ww.is_infinite() || *ww < 0.0 {
                return None;
            }
        }
        Some(Weights(Vec::from(w)))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl std::ops::Index<usize> for Weights {
    type Output = f64;
    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}
