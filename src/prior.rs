//! Prior distribution definitions and operations.
//!
//! This module provides common prior distributions used in Bayesian inference,
//! including uniform, categorical, and custom priors.

/// A prior distribution over a discrete set of hypotheses.
#[derive(Debug, Clone)]
pub struct DiscretePrior {
    /// Probability for each hypothesis.
    probs: Vec<f64>,
}

impl DiscretePrior {
    /// Create a new discrete prior from a slice of probabilities.
    ///
    /// The probabilities are automatically normalized.
    pub fn new(probs: &[f64]) -> Self {
        assert!(!probs.is_empty(), "Prior must have at least one element");
        let sum: f64 = probs.iter().sum();
        assert!(sum > 0.0, "Prior probabilities must sum to a positive value");
        let normalized: Vec<f64> = probs.iter().map(|p| p / sum).collect();
        Self { probs: normalized }
    }

    /// Create a uniform prior over `n` hypotheses.
    pub fn uniform(n: usize) -> Self {
        assert!(n > 0, "Must have at least one hypothesis");
        let p = 1.0 / n as f64;
        Self { probs: vec![p; n] }
    }

    /// Create a Jeffreys-like prior (proportional to sqrt of values).
    pub fn jeffreys(values: &[f64]) -> Self {
        let sqrt_vals: Vec<f64> = values.iter().map(|v| v.sqrt()).collect();
        Self::new(&sqrt_vals)
    }

    /// Get the probability of hypothesis at index `i`.
    pub fn prob(&self, i: usize) -> f64 {
        self.probs[i]
    }

    /// Get all probabilities as a slice.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// Number of hypotheses.
    #[must_use]
    pub fn len(&self) -> usize {
        self.probs.len()
    }

    /// Whether the prior is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.probs.is_empty()
    }

    /// Shannon entropy of the prior distribution.
    pub fn entropy(&self) -> f64 {
        -self.probs.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_prior() {
        let prior = DiscretePrior::uniform(4);
        for i in 0..4 {
            assert!((prior.prob(i) - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_custom_prior_normalized() {
        let prior = DiscretePrior::new(&[2.0, 3.0, 5.0]);
        assert!((prior.prob(0) - 0.2).abs() < 1e-10);
        assert!((prior.prob(1) - 0.3).abs() < 1e-10);
        assert!((prior.prob(2) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_prior_sums_to_one() {
        let prior = DiscretePrior::new(&[1.0, 2.0, 3.0, 4.0]);
        let sum: f64 = prior.probs().iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_entropy() {
        let prior = DiscretePrior::uniform(2);
        let expected = (2.0_f64).ln(); // ln(2)
        assert!((prior.entropy() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_deterministic_entropy_zero() {
        let prior = DiscretePrior::new(&[1.0, 0.0]);
        assert!(prior.entropy().abs() < 1e-10);
    }

    #[test]
    fn test_len() {
        let prior = DiscretePrior::uniform(5);
        assert_eq!(prior.len(), 5);
    }
}
