//! Conjugate prior families for efficient Bayesian updating.
//!
//! This module implements common conjugate prior families:
//!
//! - **Beta-Binomial**: For binary/binary-like observations
//! - **Dirichlet-Multinomial**: For categorical observations

/// Beta distribution as conjugate prior for Binomial likelihood.
///
/// The Beta(α, β) distribution is conjugate to the Binomial distribution.
/// After observing `k` successes in `n` trials, the posterior is Beta(α+k, β+n-k).
#[derive(Debug, Clone)]
pub struct BetaBinomial {
    /// Shape parameter α (pseudo-count of successes)
    pub alpha: f64,
    /// Shape parameter β (pseudo-count of failures)
    pub beta: f64,
}

impl BetaBinomial {
    /// Create a new Beta-Binomial conjugate prior.
    ///
    /// # Arguments
    /// * `alpha` - Shape parameter α > 0
    /// * `beta` - Shape parameter β > 0
    pub fn new(alpha: f64, beta: f64) -> Self {
        assert!(alpha > 0.0 && beta > 0.0, "Parameters must be positive");
        Self { alpha, beta }
    }

    /// Create a uniform prior: Beta(1, 1).
    pub fn uniform() -> Self {
        Self::new(1.0, 1.0)
    }

    /// Create a Jeffreys prior: Beta(0.5, 0.5).
    pub fn jeffreys() -> Self {
        Self::new(0.5, 0.5)
    }

    /// Update the prior with binomial observations.
    ///
    /// # Arguments
    /// * `successes` - Number of successes observed
    /// * `failures` - Number of failures observed
    pub fn update(&self, successes: u64, failures: u64) -> Self {
        Self::new(
            self.alpha + successes as f64,
            self.beta + failures as f64,
        )
    }

    /// Mean of the Beta distribution: α / (α + β).
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Variance of the Beta distribution.
    pub fn variance(&self) -> f64 {
        let a = self.alpha;
        let b = self.beta;
        (a * b) / ((a + b).powi(2) * (a + b + 1.0))
    }

    /// Mode of the Beta distribution (requires α, β > 1).
    pub fn mode(&self) -> f64 {
        assert!(self.alpha > 1.0 && self.beta > 1.0, "Mode requires α, β > 1");
        (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)
    }

    /// Log of the Beta function B(α, β) = Γ(α)Γ(β)/Γ(α+β).
    /// Computed via Stirling-like log-gamma approximation.
    fn log_beta(a: f64, b: f64) -> f64 {
        Self::log_gamma(a) + Self::log_gamma(b) - Self::log_gamma(a + b)
    }

    /// Log-gamma function using Lanczos approximation.
    #[allow(clippy::excessive_precision)]
    fn log_gamma(x: f64) -> f64 {
        // Lanczos coefficients
        let g = 7.0;
        let coef = [
            0.999_999_999_999_809_9,
            676.5203681218851,
            -1259.1392167224028,
            771.323_428_777_653_1,
            -176.615_029_162_140_6,
            12.507343278686905,
            -0.13857109526572012,
            9.984_369_578_019_572e-6,
            1.5056327351493116e-7,
        ];

        if x < 0.5 {
            // Reflection formula
            let pi = std::f64::consts::PI;
            return (pi / (pi * x).sin()).ln() - Self::log_gamma(1.0 - x);
        }

        let x = x - 1.0;
        let a = coef[0];
        let t = x + g + 0.5;

        let sum = coef.iter().skip(1).enumerate().fold(a, |acc, (i, c)| {
            acc + c / (x + i as f64 + 1.0)
        });

        0.5 * (2.0 * std::f64::consts::PI).ln()
            + (t.ln() * (x + 0.5))
            - t
            + sum.ln()
    }

    /// Probability density function of the Beta distribution at point x.
    pub fn pdf(&self, x: f64) -> f64 {
        assert!((0.0..=1.0).contains(&x), "x must be in [0, 1]");
        if x == 0.0 || x == 1.0 {
            return 0.0; // For α, β > 1
        }
        let log_p = (self.alpha - 1.0) * x.ln()
            + (self.beta - 1.0) * (1.0 - x).ln()
            - Self::log_beta(self.alpha, self.beta);
        log_p.exp()
    }

    /// The predictive probability of success on the next trial.
    pub fn predictive(&self) -> f64 {
        self.mean()
    }
}

/// Dirichlet distribution as conjugate prior for Multinomial likelihood.
///
/// The Dirichlet(α₁, ..., αₖ) distribution is conjugate to the Multinomial.
#[derive(Debug, Clone)]
pub struct DirichletMultinomial {
    /// Concentration parameters αᵢ
    pub alphas: Vec<f64>,
}

impl DirichletMultinomial {
    /// Create a new Dirichlet-Multinomial conjugate prior.
    pub fn new(alphas: &[f64]) -> Self {
        assert!(!alphas.is_empty(), "Must have at least one category");
        assert!(alphas.iter().all(|&a| a > 0.0), "All α must be positive");
        Self { alphas: alphas.to_vec() }
    }

    /// Create a symmetric Dirichlet prior with all α equal.
    pub fn symmetric(k: usize, alpha: f64) -> Self {
        assert!(k > 0 && alpha > 0.0);
        Self { alphas: vec![alpha; k] }
    }

    /// Create a uniform Dirichlet prior (all α = 1).
    pub fn uniform(k: usize) -> Self {
        Self::symmetric(k, 1.0)
    }

    /// Number of categories.
    #[must_use]
    pub fn k(&self) -> usize {
        self.alphas.len()
    }

    /// Update with observed counts.
    pub fn update(&self, counts: &[u64]) -> Self {
        assert_eq!(counts.len(), self.alphas.len(), "Mismatched lengths");
        let new_alphas: Vec<f64> = self.alphas.iter()
            .zip(counts.iter())
            .map(|(&a, &c)| a + c as f64)
            .collect();
        Self { alphas: new_alphas }
    }

    /// Mean of the Dirichlet distribution for category i.
    pub fn mean(&self, i: usize) -> f64 {
        let sum: f64 = self.alphas.iter().sum();
        self.alphas[i] / sum
    }

    /// All means as a vector.
    pub fn means(&self) -> Vec<f64> {
        let sum: f64 = self.alphas.iter().sum();
        self.alphas.iter().map(|a| a / sum).collect()
    }

    /// Variance of the Dirichlet for category i.
    pub fn variance(&self, i: usize) -> f64 {
        let sum: f64 = self.alphas.iter().sum();
        let a = self.alphas[i];
        (a * (sum - a)) / (sum * sum * (sum + 1.0))
    }

    /// Predictive probability for category i on next observation.
    pub fn predictive(&self, i: usize) -> f64 {
        self.mean(i)
    }

    /// Concentration (sum of all α parameters).
    pub fn concentration(&self) -> f64 {
        self.alphas.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_uniform() {
        let beta = BetaBinomial::uniform();
        assert!((beta.mean() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_beta_update_mean() {
        let prior = BetaBinomial::uniform();
        let post = prior.update(7, 3);
        // Beta(8, 4): mean = 8/12 = 2/3
        assert!((post.mean() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_beta_update_additive() {
        let b = BetaBinomial::new(2.0, 2.0);
        let b1 = b.update(5, 5);
        let b2 = BetaBinomial::new(7.0, 7.0);
        assert!((b1.mean() - b2.mean()).abs() < 1e-10);
    }

    #[test]
    fn test_beta_sequential_update() {
        let mut b = BetaBinomial::uniform();
        b = b.update(1, 0);
        b = b.update(1, 0);
        b = b.update(0, 1);
        let b2 = BetaBinomial::uniform().update(2, 1);
        assert!((b.alpha - b2.alpha).abs() < 1e-10);
        assert!((b.beta - b2.beta).abs() < 1e-10);
    }

    #[test]
    fn test_beta_variance_decreases() {
        let b1 = BetaBinomial::uniform();
        let b2 = b1.update(50, 50);
        assert!(b2.variance() < b1.variance());
    }

    #[test]
    fn test_beta_mode() {
        let b = BetaBinomial::new(3.0, 3.0);
        assert!((b.mode() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_beta_predictive() {
        let b = BetaBinomial::new(3.0, 7.0);
        assert!((b.predictive() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_dirichlet_uniform() {
        let d = DirichletMultinomial::uniform(3);
        let means = d.means();
        for m in &means {
            assert!((m - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dirichlet_update() {
        let d = DirichletMultinomial::uniform(3);
        let d2 = d.update(&[10, 5, 5]);
        assert!(d2.mean(0) > d2.mean(1));
    }

    #[test]
    fn test_dirichlet_means_sum_to_one() {
        let d = DirichletMultinomial::new(&[1.0, 2.0, 3.0, 4.0]);
        let sum: f64 = d.means().iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dirichlet_variance() {
        let d = DirichletMultinomial::symmetric(2, 10.0);
        let v = d.variance(0);
        // α=10, sum=20: var = 10*10/(20*20*21) = 100/8400
        assert!((v - 100.0 / 8400.0).abs() < 1e-10);
    }

    #[test]
    fn test_dirichlet_concentration() {
        let d = DirichletMultinomial::new(&[1.0, 2.0, 3.0]);
        assert!((d.concentration() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_dirichlet_predictive() {
        let d = DirichletMultinomial::new(&[2.0, 3.0]);
        assert!((d.predictive(0) - 0.4).abs() < 1e-10);
    }
}
