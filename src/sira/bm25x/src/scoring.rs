// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Method {
    Lucene,
    Robertson,
    Atire,
    BM25L,
    BM25Plus,
}

impl Method {
    pub fn to_id(self) -> u8 {
        match self {
            Method::Lucene => 0,
            Method::Robertson => 1,
            Method::Atire => 2,
            Method::BM25L => 3,
            Method::BM25Plus => 4,
        }
    }

    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(Method::Lucene),
            1 => Some(Method::Robertson),
            2 => Some(Method::Atire),
            3 => Some(Method::BM25L),
            4 => Some(Method::BM25Plus),
            _ => None,
        }
    }
}

/// Compute the IDF component for a given method.
/// `n` = total number of documents, `df` = document frequency of the term.
#[inline]
pub fn idf(method: Method, n: u32, df: u32) -> f32 {
    let n = n as f64;
    let df = df as f64;
    let val = match method {
        Method::Lucene => (1.0 + (n - df + 0.5) / (df + 0.5)).ln(),
        Method::Robertson => {
            let raw = ((n - df + 0.5) / (df + 0.5)).ln();
            if raw < 0.0 {
                0.0
            } else {
                raw
            }
        }
        Method::Atire => (n / df).ln(),
        Method::BM25L => ((n + 1.0) / (df + 0.5)).ln(),
        Method::BM25Plus => ((n + 1.0) / df).ln(),
    };
    val as f32
}

/// Compute the full BM25 score for a single term occurrence.
/// `tf` = term frequency in the document, `dl` = document length,
/// `avgdl` = average document length, `idf_val` = precomputed IDF.
/// Scoring parameters bundled together.
#[derive(Debug, Clone, Copy)]
pub struct ScoringParams {
    pub k1: f32,
    pub b: f32,
    pub delta: f32,
    pub avgdl: f32,
}

#[inline]
pub fn score(method: Method, tf: u32, dl: u32, params: &ScoringParams, idf_val: f32) -> f32 {
    let tf = tf as f32;
    let dl = dl as f32;
    let ScoringParams {
        k1,
        b,
        delta,
        avgdl,
    } = *params;
    let norm = 1.0 - b + b * dl / avgdl;

    let tfc = match method {
        Method::Lucene | Method::Robertson => tf / (k1 * norm + tf),
        Method::Atire => (tf * (k1 + 1.0)) / (tf + k1 * norm),
        Method::BM25L => {
            let c = tf / norm;
            ((k1 + 1.0) * (c + delta)) / (k1 + c + delta)
        }
        Method::BM25Plus => ((k1 + 1.0) * tf) / (k1 * norm + tf) + delta,
    };

    idf_val * tfc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idf_lucene() {
        let val = idf(Method::Lucene, 100, 10);
        // ln(1 + (100 - 10 + 0.5) / (10 + 0.5)) = ln(1 + 90.5/10.5) = ln(9.619...)
        assert!((val - 2.2624).abs() < 0.01, "got {}", val);
    }

    #[test]
    fn test_idf_robertson_clamps() {
        // When df > n/2, robertson IDF could go negative but should clamp to 0
        let val = idf(Method::Robertson, 10, 8);
        assert!(val >= 0.0, "Robertson IDF should be >= 0, got {}", val);
    }

    #[test]
    fn test_score_lucene_basic() {
        let idf_val = idf(Method::Lucene, 100, 10);
        let params = ScoringParams {
            k1: 1.5,
            b: 0.75,
            delta: 0.5,
            avgdl: 40.0,
        };
        let s = score(Method::Lucene, 3, 50, &params, idf_val);
        assert!(s > 0.0);
    }

    #[test]
    fn test_all_methods_positive() {
        for method in [
            Method::Lucene,
            Method::Robertson,
            Method::Atire,
            Method::BM25L,
            Method::BM25Plus,
        ] {
            let idf_val = idf(method, 1000, 50);
            let params = ScoringParams {
                k1: 1.5,
                b: 0.75,
                delta: 0.5,
                avgdl: 80.0,
            };
            let s = score(method, 2, 100, &params, idf_val);
            assert!(s >= 0.0, "{:?} produced negative score: {}", method, s);
        }
    }
}
