//! Naive Bayes classifier for discrete features.
//!
//! This module implements a Multinomial Naive Bayes classifier that uses
//! Dirichlet-Multinomial conjugate priors for parameter estimation.

use crate::conjugate::DirichletMultinomial;

/// A Naive Bayes classifier with discrete features.
#[derive(Debug, Clone)]
pub struct NaiveBayesClassifier {
    /// Prior probabilities for each class.
    class_priors: Vec<f64>,
    /// For each feature, for each class: Dirichlet-Multinomial posterior.
    /// feature_posteriors[feature_idx][class_idx]
    feature_posteriors: Vec<Vec<DirichletMultinomial>>,
    /// Number of classes.
    n_classes: usize,
    /// Number of features.
    n_features: usize,
}

/// Builder for training a NaiveBayesClassifier.
pub struct NaiveBayesBuilder {
    n_classes: usize,
    n_features: usize,
    n_feature_values: Vec<usize>,
    /// (class_label, feature_vector)
    data: Vec<(usize, Vec<usize>)>,
    prior_alpha: f64,
}

impl NaiveBayesBuilder {
    /// Create a new classifier builder.
    ///
    /// # Arguments
    /// * `n_classes` - Number of distinct classes
    /// * `n_features` - Number of features per instance
    /// * `n_feature_values` - Number of distinct values each feature can take
    pub fn new(n_classes: usize, n_features: usize, n_feature_values: Vec<usize>) -> Self {
        assert_eq!(n_feature_values.len(), n_features);
        Self {
            n_classes,
            n_features,
            n_feature_values,
            data: Vec::new(),
            prior_alpha: 1.0,
        }
    }

    /// Set the Dirichlet prior alpha (default: 1.0 for Laplace smoothing).
    #[must_use]
    pub fn with_prior_alpha(mut self, alpha: f64) -> Self {
        self.prior_alpha = alpha;
        self
    }

    /// Add a training example.
    #[must_use]
    pub fn add_example(mut self, class: usize, features: Vec<usize>) -> Self {
        assert_eq!(features.len(), self.n_features);
        assert!(class < self.n_classes);
        self.data.push((class, features));
        self
    }

    /// Build the trained classifier.
    pub fn build(self) -> NaiveBayesClassifier {
        let n = self.data.len() as f64;

        // Compute class priors
        let mut class_counts = vec![0usize; self.n_classes];
        for &(c, _) in &self.data {
            class_counts[c] += 1;
        }
        let class_priors: Vec<f64> = class_counts.iter()
            .map(|&c| c as f64 / n)
            .collect();

        // For each feature, for each class, count feature values
        let mut feature_posteriors = Vec::with_capacity(self.n_features);

        for f in 0..self.n_features {
            let mut class_dms = Vec::with_capacity(self.n_classes);
            for c in 0..self.n_classes {
                let mut counts = vec![0u64; self.n_feature_values[f]];
                for &(cls, ref feats) in &self.data {
                    if cls == c {
                        counts[feats[f]] += 1;
                    }
                }
                let prior = DirichletMultinomial::symmetric(self.n_feature_values[f], self.prior_alpha);
                class_dms.push(prior.update(&counts));
            }
            feature_posteriors.push(class_dms);
        }

        NaiveBayesClassifier {
            class_priors,
            feature_posteriors,
            n_classes: self.n_classes,
            n_features: self.n_features,
        }
    }
}

impl NaiveBayesClassifier {
    /// Classify a feature vector, returning the class index.
    pub fn predict(&self, features: &[usize]) -> usize {
        let probs = self.predict_proba(features);
        probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Get class probabilities for a feature vector.
    ///
    /// Returns unnormalized log probabilities converted to probabilities.
    pub fn predict_proba(&self, features: &[usize]) -> Vec<f64> {
        assert_eq!(features.len(), self.n_features);

        let mut log_probs = vec![0.0; self.n_classes];

        for (c, log_p) in log_probs.iter_mut().enumerate() {
            *log_p = self.class_priors[c].ln();
            for (f, &feat_val) in features.iter().enumerate() {
                let dm = &self.feature_posteriors[f][c];
                let p = dm.predictive(feat_val);
                *log_p += p.ln();
            }
        }

        // Convert from log to probability via softmax
        let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sums: Vec<f64> = log_probs.iter().map(|l| (l - max_log).exp()).collect();
        let total: f64 = exp_sums.iter().sum();
        exp_sums.iter().map(|e| e / total).collect()
    }

    /// Get the number of classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_binary_classifier() {
        // Simple separable data: feature 0 perfectly predicts class
        let mut builder = NaiveBayesBuilder::new(2, 2, vec![2, 2]);
        for _ in 0..10 {
            builder = builder.add_example(0, vec![0, 0]);
        }
        for _ in 0..10 {
            builder = builder.add_example(1, vec![1, 1]);
        }
        let clf = builder.build();

        assert_eq!(clf.predict(&[0, 0]), 0);
        assert_eq!(clf.predict(&[1, 1]), 1);
    }

    #[test]
    fn test_dominant_feature_classifier() {
        // Feature 0 is strongly correlated with class
        let mut builder = NaiveBayesBuilder::new(2, 2, vec![3, 2]);
        for _ in 0..20 {
            builder = builder.add_example(0, vec![0, 0]);
            builder = builder.add_example(0, vec![0, 1]);
        }
        for _ in 0..20 {
            builder = builder.add_example(1, vec![2, 0]);
            builder = builder.add_example(1, vec![2, 1]);
        }
        let clf = builder.build();

        assert_eq!(clf.predict(&[0, 0]), 0);
        assert_eq!(clf.predict(&[2, 1]), 1);
        assert_eq!(clf.predict(&[0, 1]), 0);
        assert_eq!(clf.predict(&[2, 0]), 1);
    }

    #[test]
    fn test_predict_proba_sums_to_one() {
        let clf = NaiveBayesBuilder::new(2, 2, vec![2, 2])
            .add_example(0, vec![0, 0])
            .add_example(1, vec![1, 1])
            .build();

        let probs = clf.predict_proba(&[0, 1]);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_three_class_classifier() {
        let mut builder = NaiveBayesBuilder::new(3, 1, vec![3])
            .with_prior_alpha(1.0);

        // Class 0 always has feature 0
        for _ in 0..10 {
            builder = builder.add_example(0, vec![0]);
        }
        // Class 1 always has feature 1
        for _ in 0..10 {
            builder = builder.add_example(1, vec![1]);
        }
        // Class 2 always has feature 2
        for _ in 0..10 {
            builder = builder.add_example(2, vec![2]);
        }

        let clf = builder.build();
        assert_eq!(clf.predict(&[0]), 0);
        assert_eq!(clf.predict(&[1]), 1);
        assert_eq!(clf.predict(&[2]), 2);
    }

    #[test]
    fn test_class_priors() {
        let clf = NaiveBayesBuilder::new(2, 1, vec![2])
            .add_example(0, vec![0])
            .add_example(0, vec![0])
            .add_example(1, vec![1])
            .build();

        assert_eq!(clf.n_classes(), 2);
    }

    #[test]
    fn test_weather_classifier() {
        // Simplified weather dataset
        // Features: [outlook(0=sunny,1=overcast,2=rain), wind(0=weak,1=strong)]
        let clf = NaiveBayesBuilder::new(2, 2, vec![3, 2])
            .add_example(0, vec![0, 0]) // sunny, weak -> no play
            .add_example(0, vec![0, 1]) // sunny, strong -> no play
            .add_example(1, vec![1, 0]) // overcast, weak -> play
            .add_example(1, vec![1, 1]) // overcast, strong -> play
            .add_example(1, vec![2, 0]) // rain, weak -> play
            .add_example(0, vec![2, 1]) // rain, strong -> no play
            .build();

        // Overcast should predict play
        assert_eq!(clf.predict(&[1, 0]), 1);
        assert_eq!(clf.predict(&[1, 1]), 1);
    }
}
