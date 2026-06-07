//! Bayes' theorem and basic probability operations.
//!
//! This module provides functions for computing posterior probabilities
//! using Bayes' theorem, as well as likelihood and marginal likelihood computations.

/// Compute the posterior probability using Bayes' theorem.
///
/// P(H|D) = P(D|H) * P(H) / P(D)
///
/// # Arguments
/// * `prior` - P(H): prior probability of the hypothesis
/// * `likelihood` - P(D|H): probability of data given hypothesis
/// * `marginal` - P(D): marginal probability of the data
///
/// # Panics
/// Panics if `marginal` is zero.
pub fn bayes_theorem(prior: f64, likelihood: f64, marginal: f64) -> f64 {
    assert!(marginal > 0.0, "Marginal probability must be positive");
    (likelihood * prior) / marginal
}

/// Compute the marginal likelihood P(D) by summing over all hypotheses.
///
/// P(D) = Σ P(D|H_i) * P(H_i)
///
/// # Arguments
/// * `likelihoods` - Slice of P(D|H_i) values
/// * `priors` - Slice of P(H_i) values
pub fn marginal_likelihood(likelihoods: &[f64], priors: &[f64]) -> f64 {
    assert_eq!(likelihoods.len(), priors.len(), "Slices must have equal length");
    likelihoods.iter().zip(priors.iter()).map(|(l, p)| l * p).sum()
}

/// Compute posterior probabilities for multiple hypotheses simultaneously.
///
/// Returns a vector of posterior probabilities, one per hypothesis.
///
/// # Arguments
/// * `likelihoods` - P(D|H_i) for each hypothesis
/// * `priors` - P(H_i) for each hypothesis
pub fn posterior_distribution(likelihoods: &[f64], priors: &[f64]) -> Vec<f64> {
    let marginal = marginal_likelihood(likelihoods, priors);
    likelihoods.iter().zip(priors.iter())
        .map(|(l, p)| bayes_theorem(*p, *l, marginal))
        .collect()
}

/// Compute the log-odds ratio: log(P(H|D) / P(¬H|D)).
///
/// Positive values favor the hypothesis, negative values oppose it.
///
/// # Arguments
/// * `posterior` - P(H|D)
/// * `posterior_neg` - P(¬H|D)
pub fn log_odds(posterior: f64, posterior_neg: f64) -> f64 {
    assert!(posterior > 0.0 && posterior_neg > 0.0, "Probabilities must be positive");
    (posterior / posterior_neg).ln()
}

/// Normalize a vector of unnormalized probabilities so they sum to 1.0.
pub fn normalize(probs: &[f64]) -> Vec<f64> {
    let sum: f64 = probs.iter().sum();
    assert!(sum > 0.0, "Sum must be positive for normalization");
    probs.iter().map(|p| p / sum).collect()
}

/// Sequential Bayesian update: apply multiple observations to update a prior.
///
/// Each observation updates the current belief state.
///
/// # Arguments
/// * `initial_prior` - Starting prior probabilities for each hypothesis
/// * `likelihoods_per_obs` - For each observation, P(obs|H_i) for each hypothesis
pub fn sequential_update(initial_prior: &[f64], likelihoods_per_obs: &[Vec<f64>]) -> Vec<f64> {
    let mut current = initial_prior.to_vec();
    for obs_likelihoods in likelihoods_per_obs {
        current = posterior_distribution(obs_likelihoods, &current);
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayes_theorem_basic() {
        // Classic medical test example
        let prior = 0.001; // disease prevalence
        let likelihood = 0.99; // test sensitivity
        let marginal = 0.05; // P(positive test)
        let posterior = bayes_theorem(prior, likelihood, marginal);
        assert!((posterior - 0.0198).abs() < 1e-4);
    }

    #[test]
    fn test_bayes_theorem_coin() {
        // Fair coin vs biased coin
        let prior = 0.5;
        let likelihood = 0.8; // biased coin lands heads
        let marginal = 0.65; // 0.5*0.5 + 0.5*0.8
        let posterior = bayes_theorem(prior, likelihood, marginal);
        assert!((posterior - 0.615384615).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "Marginal probability must be positive")]
    fn test_bayes_theorem_zero_marginal() {
        bayes_theorem(0.5, 0.5, 0.0);
    }

    #[test]
    fn test_marginal_likelihood() {
        let likelihoods = [0.9, 0.1];
        let priors = [0.5, 0.5];
        let m = marginal_likelihood(&likelihoods, &priors);
        assert!((m - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_posterior_distribution_sums_to_one() {
        let likelihoods = [0.3, 0.5, 0.2];
        let priors = [0.4, 0.35, 0.25];
        let posterior = posterior_distribution(&likelihoods, &priors);
        let sum: f64 = posterior.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_posterior_distribution_uniform_prior() {
        let likelihoods = [0.1, 0.3, 0.6];
        let priors = [1.0 / 3.0; 3];
        let posterior = posterior_distribution(&likelihoods, &priors);
        assert!((posterior[2] - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_log_odds() {
        let lo = log_odds(0.75, 0.25);
        assert!((lo - 1.098612289).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let probs = [2.0, 3.0, 5.0];
        let normed = normalize(&probs);
        assert!((normed[0] - 0.2).abs() < 1e-10);
        assert!((normed[1] - 0.3).abs() < 1e-10);
        assert!((normed[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_sums_to_one() {
        let probs = [1.0, 2.0, 3.0, 4.0];
        let normed = normalize(&probs);
        let sum: f64 = normed.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sequential_update_convergence() {
        // Repeated observations of a biased coin
        let prior = [0.5, 0.5];
        // Each observation: [P(heads|fair), P(heads|biased)]
        let observations = vec![
            vec![0.5, 0.9],
            vec![0.5, 0.9],
            vec![0.5, 0.9],
            vec![0.5, 0.9],
            vec![0.5, 0.9],
            vec![0.5, 0.9],
            vec![0.5, 0.9],
            vec![0.5, 0.9],
        ];
        let result = sequential_update(&prior, &observations);
        // After 8 heads observations, should strongly favor biased coin
        assert!(result[1] > 0.9);
    }
}
