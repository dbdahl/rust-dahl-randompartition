use rand::distributions::{Distribution, Uniform};
use rand::Rng;

/// The slice sampler using the stepping out and shrinkage procedures
pub fn slice_sampler<F: FnMut(f64) -> f64, R: Rng>(
    x: f64,
    mut f: F,
    w: f64,
    max_number_of_steps: u32,
    on_log_scale: bool,
    rng: &mut R,
) -> (f64, u32) {
    let w = if w <= 0.0 { f64::MIN_POSITIVE } else { w };
    let uniform = Uniform::from(0.0..1.0);
    let mut u = || uniform.sample(rng);
    let mut evaluation_counter = 0;
    let mut f_with_counter = |x: f64| {
        evaluation_counter += 1;
        f(x)
    };
    // Step 1 (slicing)
    let y = {
        let u: f64 = u();
        let fx = f_with_counter(x);
        if on_log_scale {
            u.ln() + fx
        } else {
            u * fx
        }
    };
    // Step 2 (stepping out)
    let mut l = x - u() * w;
    let mut r = l + w;
    if max_number_of_steps == 0 {
        while y < f_with_counter(l) {
            l -= w
        }
        while y < f_with_counter(r) {
            r += w
        }
    } else {
        let mut j = (u() * (max_number_of_steps as f64)).floor() as u32;
        let mut k = max_number_of_steps - 1 - j;
        while j > 0 && y < f_with_counter(l) {
            l -= w;
            j -= 1;
        }
        while k > 0 && y < f_with_counter(r) {
            r += w;
            k -= 1;
        }
    }
    // Step 3 (shrinkage)
    loop {
        let x1 = l + u() * (r - l);
        let fx1 = f_with_counter(x1);
        if y < fx1 {
            return (x1, evaluation_counter);
        }
        if x1 < x {
            l = x1;
        } else {
            r = x1;
        }
    }
}
