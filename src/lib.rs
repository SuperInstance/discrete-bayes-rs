//! # discrete-bayes-rs
//!
//! A pure-Rust library for discrete Bayesian inference, providing:
//! - Bayes' theorem computation
//! - Prior and posterior distribution handling
//! - Conjugate prior families (Beta-Binomial, Dirichlet-Multinomial)
//! - Naive Bayes classifier
//!
//! ## Example
//!
//! ```
//! use discrete_bayes_rs::bayes::bayes_theorem;
//!
//! let posterior = bayes_theorem(0.01, 0.99, 0.05);
//! assert!(posterior > 0.0 && posterior < 1.0);
//! ```

/// Bayes' theorem and basic probability operations.
pub mod bayes;

/// Prior distribution definitions and operations.
pub mod prior;

/// Posterior distribution computation and updating.
pub mod posterior;

/// Conjugate prior families (Beta-Binomial, Dirichlet-Multinomial).
pub mod conjugate;

/// Naive Bayes classifier for discrete features.
pub mod classifier;
