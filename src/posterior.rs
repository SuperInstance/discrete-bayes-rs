//! Posterior distribution computation and updating.
//!
//! This module provides utilities for computing posterior distributions
//! from priors and likelihoods, including MAP estimation.

use crate::bayes::normalize;
use crate::prior::DiscretePrior;

/// A posterior distribution resulting from Bayesian updating.
#[derive(Debug, Clone)]
pub struct DiscretePosterior {
    /// Posterior probabilities for each hypothesis.
    probs: Vec<f64>,
}

impl DiscretePosterior {
    /// Compute the posterior from a prior and likelihoods.
    ///
    /// # Arguments
    /// * `prior` - The prior distribution
    /// * `likelihoods` - P(D|H_i) for each hypothesis
    pub fn from_prior_likelihood(prior: &DiscretePrior, likelihoods: &[f64]) -> Self {
        assert_eq!(prior.len(), likelihoods.len(), "Mismatched lengths");
        let unnormalized: Vec<f64> = likelihoods.iter()
            .zip(prior.probs().iter())
            .map(|(l, p)| l * p)
            .collect();
        let probs = normalize(&unnormalized);
        Self { probs }
    }

    /// Get the probability of hypothesis at index `i`.
    pub fn prob(&self, i: usize) -> f64 {
        self.probs[i]
    }

    /// Get all posterior probabilities.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// Maximum a posteriori (MAP) estimate: the index of the most probable hypothesis.
    pub fn map(&self) -> usize {
        self.probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Update the posterior with a new observation.
    pub fn update(&self, likelihoods: &[f64]) -> Self {
        assert_eq!(self.probs.len(), likelihoods.len(), "Mismatched lengths");
        let unnormalized: Vec<f64> = likelihoods.iter()
            .zip(self.probs.iter())
            .map(|(l, p)| l * p)
            .collect();
        let probs = normalize(&unnormalized);
        Self { probs }
    }

    /// Compute the Bayes factor comparing hypothesis i to hypothesis j.
    ///
    /// BF_ij = P(H_i|D) / P(H_j|D)
    pub fn bayes_factor(&self, i: usize, j: usize) -> f64 {
        assert!(self.probs[j] > 0.0, "Denominator must be positive");
        self.probs[i] / self.probs[j]
    }

    /// Credible interval: find the smallest set of hypotheses whose
    /// cumulative probability >= `level`.
    ///
    /// Returns indices of hypotheses in the credible set.
    pub fn credible_set(&self, level: f64) -> Vec<usize> {
        let mut indexed: Vec<(usize, f64)> = self.probs.iter().enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut cumsum = 0.0;
        let mut result = Vec::new();
        for (idx, prob) in indexed {
            result.push(idx);
            cumsum += prob;
            if cumsum >= level {
                break;
            }
        }
        result.sort();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_prior() -> DiscretePrior {
        DiscretePrior::uniform(3)
    }

    #[test]
    fn test_posterior_from_uniform_prior() {
        let prior = make_prior();
        let likelihoods = [0.1, 0.3, 0.6];
        let post = DiscretePosterior::from_prior_likelihood(&prior, &likelihoods);
        assert!((post.prob(2) - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_posterior_sums_to_one() {
        let prior = DiscretePrior::new(&[0.2, 0.3, 0.5]);
        let likelihoods = [0.5, 0.4, 0.1];
        let post = DiscretePosterior::from_prior_likelihood(&prior, &likelihoods);
        let sum: f64 = post.probs().iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_map_estimate() {
        let prior = DiscretePrior::uniform(4);
        let likelihoods = [0.1, 0.2, 0.6, 0.1];
        let post = DiscretePosterior::from_prior_likelihood(&prior, &likelihoods);
        assert_eq!(post.map(), 2);
    }

    #[test]
    fn test_sequential_update() {
        let prior = DiscretePrior::uniform(2);
        let post1 = DiscretePosterior::from_prior_likelihood(&prior, &[0.5, 0.9]);
        let post2 = post1.update(&[0.5, 0.9]);
        // Should converge toward hypothesis 1
        assert!(post2.prob(1) > post1.prob(1));
    }

    #[test]
    fn test_bayes_factor() {
        let prior = DiscretePrior::uniform(2);
        let post = DiscretePosterior::from_prior_likelihood(&prior, &[0.3, 0.7]);
        let bf = post.bayes_factor(1, 0);
        assert!((bf - (0.7 / 0.3)).abs() < 1e-6);
    }

    #[test]
    fn test_credible_set_95() {
        let prior = DiscretePrior::uniform(3);
        let likelihoods = [0.1, 0.3, 0.6];
        let post = DiscretePosterior::from_prior_likelihood(&prior, &likelihoods);
        let cs = post.credible_set(0.95);
        // 0.6 + 0.3 = 0.9, so need all three for 0.95
        assert_eq!(cs.len(), 3);
    }

    #[test]
    fn test_credible_set_50() {
        let prior = DiscretePrior::uniform(3);
        let likelihoods = [0.05, 0.15, 0.8];
        let post = DiscretePosterior::from_prior_likelihood(&prior, &likelihoods);
        let cs = post.credible_set(0.5);
        // 0.8 alone covers 50%
        assert_eq!(cs, vec![2]);
    }
}
