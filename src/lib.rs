//! # Cashflow Analysis
//!
//! The `cashflow-analysis` crate provides a comprehensive toolkit for defining, simulating, and analyzing 
//! cashflow events across multiple quarters. It supports both deterministic cashflows and stochastic variations 
//! via common probability distributions.
//!
//! ## Features
//!
//! - **Flexible Event Definition:** Define cashflow events using the [`CashflowEvent`] enum.
//! - **Sequential Scheduling:** Compose events into a [`CashflowSchedule`] where each event represents a quarter.
//! - **Monte Carlo Simulation:** Run simulations to obtain statistical summaries of cumulative cashflows, including
//!   sample mean, sample variance, minimum, and maximum values.
//! - **Composable Transformations:** Easily scale, offset, and combine schedules using built-in helper methods.
//!
//! ## Examples
//!
//! Simulate a simple cashflow schedule:
//!
//! ```rust
//! use cashflow_analysis::{CashflowSchedule, CashflowEvent, StochasticCashflowEvent};
//! use rand::thread_rng;
//!
//! let mut schedule = CashflowSchedule::new();
//! schedule.add_event(CashflowEvent::Deterministic(100.0));
//! schedule.add_event(CashflowEvent::Stochastic(
//!     StochasticCashflowEvent::Uniform { min: 10.0, max: 20.0 }
//! ));
//!
//! let result = schedule.run_monte_carlo(10_000, thread_rng());
//!
//! for stat in result.quarter_stats {
//!     println!(
//!         "Quarter {}: Mean: {:.2}, Variance: {:.2}, Min: {:.2}, Max: {:.2}",
//!         stat.quarter, stat.mean, stat.variance, stat.min, stat.max
//!     );
//! }
//! ```
//!
//! ## Limitations
//!
//! - Invalid parameters (e.g., negative standard deviation) trigger panics.
//! - For high precision, a large number of simulation trials is recommended.
//!
//! ## Future Work
//!
//! Future improvements may include robust error handling and support for additional probability distributions.
//!
//! ## Licensing & Repository
//!
//! See the LICENSE file for details. Visit [GitHub](https://github.com/SamuelSchlesinger/cashflow-analysis) for more information.

use rand::Rng;
use rand::distr::{Distribution, weighted::WeightedIndex, Uniform};
use rand_distr::{Normal, LogNormal, Exp};

/// Represents a cashflow event, which may be either deterministic or stochastic.
///
/// # Variants
///
/// - `Deterministic(f64)`: A fixed cashflow value.
/// - `Stochastic(StochasticCashflowEvent)`: A cashflow defined by a probabilistic model.
pub enum CashflowEvent {
    Deterministic(f64),
    Stochastic(StochasticCashflowEvent),
}

pub enum StochasticCashflowEvent {
    /// Normal (Gaussian) distribution with probability density function:
    /// f(x) = (1/(σ√(2π))) * exp(-(x-μ)²/(2σ²))
    /// where μ is the mean and σ is the standard deviation
    Normal {
        mean: f64,
        std_dev: f64,
    },
    /// Log-normal distribution with probability density function:
    /// f(x) = (1/(xσ√(2π))) * exp(-(ln(x)-μ)²/(2σ²))
    /// where μ is the location parameter and σ is the scale parameter
    LogNormal {
        location: f64,
        scale: f64,
    },
    /// Continuous uniform distribution with probability density function:
    /// f(x) = 1/(b-a) for a ≤ x ≤ b
    /// where a is the minimum value and b is the maximum value
    Uniform {
        min: f64,
        max: f64,
    },
    /// Discrete uniform distribution with probability mass function:
    /// P(X = xᵢ) = pᵢ
    /// where xᵢ are the possible values and pᵢ are their corresponding probabilities
    /// with Σpᵢ = 1
    DiscreteUniform {
        values: Vec<(f64, f64)>,
    },
    /// Exponential distribution with probability density function:
    /// f(x) = λe^(-λx) for x ≥ 0
    /// where λ is the rate parameter
    Exponential {
        rate: f64,
    },
}

/// A schedule of cashflow events occurring sequentially by quarter.
/// 
/// Each event in the schedule represents the cashflow for a particular quarter (1-indexed).
/// Use helper methods like [`scale`], [`offset`], [`append_schedule`], and [`combine`] to transform
/// or compose schedules.
pub struct CashflowSchedule {
    /// Vector of cashflow events for each quarter.
    pub events: Vec<CashflowEvent>,
}

impl CashflowSchedule {
    /// Creates a new, empty cashflow schedule.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cashflow_analysis::CashflowSchedule;
    /// let schedule = CashflowSchedule::new();
    /// assert_eq!(schedule.events.len(), 0);
    /// ```
    pub fn new() -> Self {
        CashflowSchedule {
            events: Vec::new()
        }
    }

    /// Creates a new cashflow schedule with predefined events.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cashflow_analysis::{CashflowSchedule, CashflowEvent};
    /// let schedule = CashflowSchedule::with_events(vec![CashflowEvent::Deterministic(100.0)]);
    /// assert_eq!(schedule.events.len(), 1);
    /// ```
    pub fn with_events(events: Vec<CashflowEvent>) -> Self {
        CashflowSchedule { events }
    }

    /// Returns the number of quarters (events) in the schedule.
    pub fn num_quarters(&self) -> usize {
        self.events.len()
    }

    /// Adds a cashflow event to the schedule.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cashflow_analysis::{CashflowSchedule, CashflowEvent};
    /// let mut schedule = CashflowSchedule::new();
    /// schedule.add_event(CashflowEvent::Deterministic(100.0));
    /// assert_eq!(schedule.num_quarters(), 1);
    /// ```
    pub fn add_event(&mut self, event: CashflowEvent) {
        self.events.push(event);
    }

    /// Returns a new schedule with all events scaled by a constant multiplier.
    ///
    /// For deterministic events this multiplies the cashflow directly. For stochastic events,
    /// the relevant parameters (such as mean, min, and max) are scaled. (Note that for some events,
    /// such as Exponential, scaling may not be directly meaningful.)
    pub fn scale(&self, factor: f64) -> CashflowSchedule {
        let events = self.events.iter().map(|ev| {
            match ev {
                CashflowEvent::Deterministic(val) => CashflowEvent::Deterministic(val * factor),
                CashflowEvent::Stochastic(stoch) => {
                    let new_stoch = match stoch {
                        StochasticCashflowEvent::Normal { mean, std_dev } => {
                            StochasticCashflowEvent::Normal { mean: mean * factor, std_dev: std_dev * factor }
                        },
                        StochasticCashflowEvent::LogNormal { location, scale } => {
                            StochasticCashflowEvent::LogNormal { location: location * factor, scale: scale * factor }
                        },
                        StochasticCashflowEvent::Uniform { min, max } => {
                            StochasticCashflowEvent::Uniform { min: min * factor, max: max * factor }
                        },
                        StochasticCashflowEvent::DiscreteUniform { values } => {
                            let scaled: Vec<(f64, f64)> = values.iter().map(|(val, prob)| (val * factor, *prob)).collect();
                            StochasticCashflowEvent::DiscreteUniform { values: scaled }
                        },
                        StochasticCashflowEvent::Exponential { rate } => {
                            // For exponential events, scaling the output might be applied after sampling.
                            // Leaving the rate unchanged here.
                            StochasticCashflowEvent::Exponential { rate: *rate }
                        },
                    };
                    CashflowEvent::Stochastic(new_stoch)
                }
            }
        }).collect();
        CashflowSchedule::with_events(events)
    }

    /// Returns a new schedule with all deterministic events offset by a constant value.
    ///
    /// For stochastic events (e.g. Normal, Uniform), this applies a shift to the relevant parameter.
    pub fn offset(&self, shift: f64) -> CashflowSchedule {
        let events = self.events.iter().map(|ev| {
            match ev {
                CashflowEvent::Deterministic(val) => CashflowEvent::Deterministic(val + shift),
                CashflowEvent::Stochastic(stoch) => {
                    let new_stoch = match stoch {
                        StochasticCashflowEvent::Normal { mean, std_dev } => {
                            StochasticCashflowEvent::Normal { mean: mean + shift, std_dev: *std_dev }
                        },
                        StochasticCashflowEvent::LogNormal { location, scale } => {
                            StochasticCashflowEvent::LogNormal { location: location + shift, scale: *scale }
                        },
                        StochasticCashflowEvent::Uniform { min, max } => {
                            StochasticCashflowEvent::Uniform { min: min + shift, max: max + shift }
                        },
                        StochasticCashflowEvent::DiscreteUniform { values } => {
                            let shifted: Vec<(f64, f64)> = values.iter().map(|(val, prob)| (val + shift, *prob)).collect();
                            StochasticCashflowEvent::DiscreteUniform { values: shifted }
                        },
                        StochasticCashflowEvent::Exponential { rate } => {
                            // Offsetting an exponential distribution is typically not done.
                            StochasticCashflowEvent::Exponential { rate: *rate }
                        },
                    };
                    CashflowEvent::Stochastic(new_stoch)
                }
            }
        }).collect();
        CashflowSchedule::with_events(events)
    }

    /// Appends all events from another schedule to this schedule.
    ///
    /// This method extends the current schedule with events from `other`,
    /// preserving the sequential order.
    pub fn append_schedule(&mut self, other: CashflowSchedule) {
        self.events.extend(other.events);
    }

    /// Combines two schedules with matching quarters by performing element-wise addition
    /// of deterministic events.
    ///
    /// Returns `Some(schedule)` where each event is the sum of corresponding deterministic events,
    /// or `None` if the schedules have different lengths or if any pair of events are not both deterministic.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cashflow_analysis::{CashflowSchedule, CashflowEvent};
    /// let mut schedule1 = CashflowSchedule::new();
    /// schedule1.add_event(CashflowEvent::Deterministic(100.0));
    /// schedule1.add_event(CashflowEvent::Deterministic(200.0));
    ///
    /// let mut schedule2 = CashflowSchedule::new();
    /// schedule2.add_event(CashflowEvent::Deterministic(10.0));
    /// schedule2.add_event(CashflowEvent::Deterministic(20.0));
    ///
    /// let combined = schedule1.combine(&schedule2).unwrap();
    /// # if let CashflowEvent::Deterministic(val) = combined.events[0] {
    /// #     assert_eq!(val, 110.0);
    /// # }
    /// ```
    pub fn combine(&self, other: &CashflowSchedule) -> Option<CashflowSchedule> {
        if self.num_quarters() != other.num_quarters() {
            return None;
        }
        let combined_events = self.events.iter().zip(other.events.iter()).map(|(a, b)| {
            match (a, b) {
                (CashflowEvent::Deterministic(x), CashflowEvent::Deterministic(y)) =>
                    Some(CashflowEvent::Deterministic(x + y)),
                _ => return None,
            }
        }).collect::<Option<Vec<_>>>()?;
        Some(CashflowSchedule::with_events(combined_events))
    }

    /// Executes a Monte Carlo simulation on the cashflow schedule over a specified number
    /// of trials.
    ///
    /// This method sequentially simulates the cumulative cashflow across quarters for each trial,
    /// then computes summary statistics for every quarter:
    ///
    /// - **Mean:** Sample mean of cumulative cashflows.
    /// - **Variance:** Sample variance (using n-1 as the denominator).
    /// - **Min/Max:** The minimum and maximum cumulative values observed.
    ///
    /// # Parameters
    ///
    /// - `num_trials`: The number of simulation trials to run.
    /// - `rng`: A random number generator implementing the [`Rng`] trait.
    ///
    /// # Returns
    ///
    /// A [`SimulationResult`] containing per-quarter summary statistics.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cashflow_analysis::{CashflowSchedule, CashflowEvent, StochasticCashflowEvent};
    /// use rand::thread_rng;
    /// 
    /// let mut schedule = CashflowSchedule::new();
    /// schedule.add_event(CashflowEvent::Deterministic(100.0));
    /// schedule.add_event(CashflowEvent::Stochastic(
    ///     StochasticCashflowEvent::Uniform { min: 10.0, max: 20.0 }
    /// ));
    /// 
    /// let result = schedule.run_monte_carlo(10_000, thread_rng());
    /// 
    /// for stat in result.quarter_stats {
    ///     println!(
    ///         "Quarter {}: Mean: {:.2}, Variance: {:.2}, Min: {:.2}, Max: {:.2}",
    ///         stat.quarter, stat.mean, stat.variance, stat.min, stat.max
    ///     );
    /// }
    /// ```
    pub fn run_monte_carlo(&self, num_trials: usize, mut rng: impl Rng) -> SimulationResult {
        let num_quarters = self.num_quarters();
        // Prepare a vector to record cumulative cashflow outcomes per quarter across trials.
        let mut outcomes: Vec<Vec<f64>> = vec![Vec::with_capacity(num_trials); num_quarters];
        // Run simulation trials.
        for _ in 0..num_trials {
            let mut cumulative = 0.0;
            for (q, event) in self.events.iter().enumerate() {
                let value = simulate_cashflow_event(event, &mut rng);
                cumulative += value;
                // Record the cumulative value for the quarter.
                outcomes[q].push(cumulative);
            }
        }

        // Compute statistics for each quarter.
        let quarter_stats = outcomes
            .into_iter()
            .enumerate()
            .map(|(q, values)| {
                let n = values.len();
                let mean = values.iter().sum::<f64>() / n as f64;
                let variance = if n > 1 {
                    values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / ((n - 1) as f64)
                } else {
                    0.0
                };
                let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                QuarterStatistics {
                    quarter: q + 1,
                    mean,
                    variance,
                    min,
                    max,
                }
            })
            .collect();

        SimulationResult { quarter_stats }
    }
}

// --------------------------------------------------------------------
// Helper function to simulate a single cashflow event
fn simulate_cashflow_event<R: Rng + ?Sized>(event: &CashflowEvent, rng: &mut R) -> f64 {
    match event {
        CashflowEvent::Deterministic(value) => *value,
        CashflowEvent::Stochastic(stoch_event) => {
            match stoch_event {
                StochasticCashflowEvent::Normal { mean, std_dev } => {
                    let normal = Normal::new(*mean, *std_dev).unwrap();
                    normal.sample(rng)
                },
                StochasticCashflowEvent::LogNormal { location, scale } => {
                    let lognormal = LogNormal::new(*location, *scale).unwrap();
                    lognormal.sample(rng)
                },
                StochasticCashflowEvent::Uniform { min, max } => {
                    let uniform = Uniform::new_inclusive(*min, *max).unwrap();
                    uniform.sample(rng)
                },
                StochasticCashflowEvent::DiscreteUniform { values } => {
                    let weights: Vec<f64> = values.iter().map(|&(_val, prob)| prob).collect();
                    let dist = WeightedIndex::new(&weights).unwrap();
                    let index = dist.sample(rng);
                    values[index].0
                },
                StochasticCashflowEvent::Exponential { rate } => {
                    let exp = Exp::new(*rate).unwrap();
                    exp.sample(rng)
                },
            }
        }
    }
}

// --------------------------------------------------------------------
// Structs to store simulation statistics
#[derive(Debug)]
pub struct QuarterStatistics {
    pub quarter: usize,   // Quarter number (1-indexed)
    pub mean: f64,        // Mean cumulative cashflow for this quarter
    pub variance: f64,    // Variance of the cumulative cashflow
    pub min: f64,         // Minimum cumulative cashflow observed
    pub max: f64,         // Maximum cumulative cashflow observed
}

#[derive(Debug)]
pub struct SimulationResult {
    pub quarter_stats: Vec<QuarterStatistics>,
}

/// A builder for constructing a [`CashflowSchedule`] using a fluent API.
///
/// This builder provides a concise and readable way to assemble a schedule:
///
/// ```rust
/// use cashflow_analysis::{CashflowScheduleBuilder, CashflowEvent, StochasticCashflowEvent};
/// 
/// let schedule = CashflowScheduleBuilder::new()
///     .add_event(CashflowEvent::Deterministic(100.0))
///     .add_event(CashflowEvent::Stochastic(
///         StochasticCashflowEvent::Uniform { min: 10.0, max: 20.0 }
///     ))
///     .build();
/// 
/// assert_eq!(schedule.num_quarters(), 2);
/// ```
/// 
/// This builder is particularly useful when constructing schedules from dynamic data.
pub struct CashflowScheduleBuilder {
    events: Vec<CashflowEvent>,
}

impl CashflowScheduleBuilder {
    /// Creates a new schedule builder.
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Adds a cashflow event to the builder.
    pub fn add_event(mut self, event: CashflowEvent) -> Self {
        self.events.push(event);
        self
    }

    /// Finalizes the builder and creates a [`CashflowSchedule`].
    pub fn build(self) -> CashflowSchedule {
        CashflowSchedule::with_events(self.events)
    }
}

impl Default for CashflowScheduleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn test_deterministic_simulation() {
        // Create a schedule with two deterministic events.
        let mut schedule = CashflowSchedule::new();
        schedule.add_event(CashflowEvent::Deterministic(100.0));
        schedule.add_event(CashflowEvent::Deterministic(50.0));

        // Use a deterministic StdRng seeded with a constant.
        let rng = StdRng::seed_from_u64(42);
        let result = schedule.run_monte_carlo(100, rng);

        // We expect each trial to yield [100.0, 150.0].
        assert_eq!(result.quarter_stats.len(), 2);

        let stat_q1 = &result.quarter_stats[0];
        assert!((stat_q1.mean - 100.0).abs() < f64::EPSILON, "Expected Q1 mean 100.0, got {}", stat_q1.mean);
        assert!((stat_q1.variance - 0.0).abs() < f64::EPSILON);
        assert!((stat_q1.min - 100.0).abs() < f64::EPSILON);
        assert!((stat_q1.max - 100.0).abs() < f64::EPSILON);

        let stat_q2 = &result.quarter_stats[1];
        assert!((stat_q2.mean - 150.0).abs() < f64::EPSILON, "Expected Q2 mean 150.0, got {}", stat_q2.mean);
        assert!((stat_q2.variance - 0.0).abs() < f64::EPSILON);
        assert!((stat_q2.min - 150.0).abs() < f64::EPSILON);
        assert!((stat_q2.max - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_schedule_simulation() {
        let schedule = CashflowSchedule::new();
        let rng = StdRng::seed_from_u64(123);
        let result = schedule.run_monte_carlo(100, rng);
        // With no events, the simulation result should contain no quarter statistics.
        assert_eq!(result.quarter_stats.len(), 0);
    }

    #[test]
    fn test_uniform_simulation_range() {
        // Create a schedule with a uniform random event for Q1,
        // followed by a deterministic event for Q2.
        let mut schedule = CashflowSchedule::new();
        schedule.add_event(CashflowEvent::Stochastic(StochasticCashflowEvent::Uniform { min: 10.0, max: 20.0 }));
        schedule.add_event(CashflowEvent::Deterministic(30.0));

        let rng = StdRng::seed_from_u64(789);
        let result = schedule.run_monte_carlo(10_000, rng);

        // For Q1, the simulated values should lie in the range [10, 20].
        let stat_q1 = &result.quarter_stats[0];
        assert!(stat_q1.mean > 12.0 && stat_q1.mean < 18.0, "Uniform event mean expected between 12 and 18, got {}", stat_q1.mean);
        assert!(stat_q1.min >= 10.0 && stat_q1.max <= 20.0, "Uniform event range expected between [10,20], got [{}, {}]", stat_q1.min, stat_q1.max);

        // For Q2, the cumulative value is the Q1 value plus a deterministic 30.
        let stat_q2 = &result.quarter_stats[1];
        assert!((stat_q2.mean - (stat_q1.mean + 30.0)).abs() < 1e-6);
        assert!((stat_q2.min - (stat_q1.min + 30.0)).abs() < 1e-6);
        assert!((stat_q2.max - (stat_q1.max + 30.0)).abs() < 1e-6);
    }

    #[test]
    fn test_discrete_uniform_simulation() {
        // Use a discrete uniform distribution with two outcomes.
        let mut schedule = CashflowSchedule::new();
        // Options: 100.0 with probability 0.3, and 200.0 with probability 0.7.
        schedule.add_event(CashflowEvent::Stochastic(StochasticCashflowEvent::DiscreteUniform {
            values: vec![(100.0, 0.3), (200.0, 0.7)]
        }));

        let rng = StdRng::seed_from_u64(555);
        let result = schedule.run_monte_carlo(10_000, rng);

        let stat_q1 = &result.quarter_stats[0];
        // Check that the mean is between 100 and 200, and the min/max must be one of these values.
        assert!(stat_q1.mean >= 100.0 && stat_q1.mean <= 200.0, "Discrete uniform mean out of range: {}", stat_q1.mean);
        assert!( (stat_q1.min - 100.0).abs() < f64::EPSILON || (stat_q1.min - 200.0).abs() < f64::EPSILON,
                "Discrete uniform min unexpected: {}", stat_q1.min);
        assert!( (stat_q1.max - 100.0).abs() < f64::EPSILON || (stat_q1.max - 200.0).abs() < f64::EPSILON,
                "Discrete uniform max unexpected: {}", stat_q1.max);
    }

    #[test]
    fn test_normal_simulation() {
        // Create a schedule with a normal random event.
        let mut schedule = CashflowSchedule::new();
        schedule.add_event(CashflowEvent::Stochastic(StochasticCashflowEvent::Normal {
            mean: 50.0,
            std_dev: 5.0,
        }));

        let rng = StdRng::seed_from_u64(999);
        let result = schedule.run_monte_carlo(10_000, rng);

        let stat_q1 = &result.quarter_stats[0];
        // Expect the mean to be around 50.
        assert!((stat_q1.mean - 50.0).abs() < 1.0, "Normal event mean too far from expected: {}", stat_q1.mean);
        // Variance should be roughly 25.
        assert!(stat_q1.variance > 20.0 && stat_q1.variance < 30.0, "Normal event variance out of range: {}", stat_q1.variance);
    }

    #[test]
    fn test_exponential_simulation() {
        // Create a schedule with an exponential event (rate = 0.1, theoretical mean = 10).
        let mut schedule = CashflowSchedule::new();
        schedule.add_event(CashflowEvent::Stochastic(StochasticCashflowEvent::Exponential { rate: 0.1 }));

        let rng = StdRng::seed_from_u64(2021);
        let result = schedule.run_monte_carlo(10_000, rng);

        let stat_q1 = &result.quarter_stats[0];
        // Ensure the mean is close to 10.
        assert!((stat_q1.mean - 10.0).abs() < 1.0, "Exponential event mean not as expected: {}", stat_q1.mean);
    }

    #[test]
    fn test_scale_offset() {
        // Create a schedule with one deterministic and one stochastic (Uniform) event.
        let mut schedule = CashflowSchedule::new();
        schedule.add_event(CashflowEvent::Deterministic(100.0));
        schedule.add_event(CashflowEvent::Stochastic(
            StochasticCashflowEvent::Uniform { min: 10.0, max: 20.0 }
        ));

        // Test scaling
        let scaled_schedule = schedule.scale(2.0);
        // First event: 100.0 becomes 200.0
        if let CashflowEvent::Deterministic(val) = scaled_schedule.events[0] {
            assert!((val - 200.0).abs() < f64::EPSILON, "Expected 200.0, got {}", val);
        } else {
            panic!("Expected a Deterministic event");
        }
        // Second event: Uniform (10,20) becomes Uniform (20,40)
        if let CashflowEvent::Stochastic(StochasticCashflowEvent::Uniform { min, max }) = &scaled_schedule.events[1] {
            assert!((*min - 20.0).abs() < f64::EPSILON, "Expected scaled min 20, got {}", min);
            assert!((*max - 40.0).abs() < f64::EPSILON, "Expected scaled max 40, got {}", max);
        } else {
            panic!("Expected a Uniform event");
        }

        // Test offsetting
        let offset_schedule = schedule.offset(10.0);
        if let CashflowEvent::Deterministic(val) = offset_schedule.events[0] {
            assert!((val - 110.0).abs() < f64::EPSILON, "Expected 110.0, got {}", val);
        } else {
            panic!("Expected a Deterministic event");
        }
        if let CashflowEvent::Stochastic(StochasticCashflowEvent::Uniform { min, max }) = &offset_schedule.events[1] {
            assert!((*min - 20.0).abs() < f64::EPSILON, "Expected offset min 20, got {}", min);
            assert!((*max - 30.0).abs() < f64::EPSILON, "Expected offset max 30, got {}", max);
        } else {
            panic!("Expected a Uniform event");
        }
    }

    #[test]
    fn test_append_schedule() {
        let mut schedule1 = CashflowSchedule::new();
        schedule1.add_event(CashflowEvent::Deterministic(100.0));

        let mut schedule2 = CashflowSchedule::new();
        schedule2.add_event(CashflowEvent::Deterministic(50.0));

        let len1 = schedule1.events.len();
        let len2 = schedule2.events.len();

        schedule1.append_schedule(schedule2);
        assert_eq!(schedule1.events.len(), len1 + len2);
    }

    #[test]
    fn test_combine_success() {
        // Two schedules with matching lengths and only deterministic events.
        let mut schedule1 = CashflowSchedule::new();
        schedule1.add_event(CashflowEvent::Deterministic(100.0));
        schedule1.add_event(CashflowEvent::Deterministic(200.0));

        let mut schedule2 = CashflowSchedule::new();
        schedule2.add_event(CashflowEvent::Deterministic(10.0));
        schedule2.add_event(CashflowEvent::Deterministic(20.0));

        let combined = schedule1.combine(&schedule2);
        assert!(combined.is_some());
        let comb = combined.unwrap();
        assert_eq!(comb.events.len(), 2);

        if let CashflowEvent::Deterministic(val) = comb.events[0] {
            assert!((val - 110.0).abs() < f64::EPSILON, "Expected 110.0, got {}", val);
        } else {
            panic!("Expected a deterministic event");
        }
        if let CashflowEvent::Deterministic(val) = comb.events[1] {
            assert!((val - 220.0).abs() < f64::EPSILON, "Expected 220.0, got {}", val);
        } else {
            panic!("Expected a deterministic event");
        }
    }

    #[test]
    fn test_combine_failure() {
        // Test failure for schedules with different lengths.
        let mut schedule1 = CashflowSchedule::new();
        schedule1.add_event(CashflowEvent::Deterministic(100.0));
        schedule1.add_event(CashflowEvent::Deterministic(200.0));

        let mut schedule2 = CashflowSchedule::new();
        schedule2.add_event(CashflowEvent::Deterministic(10.0));

        assert!(schedule1.combine(&schedule2).is_none());

        // Test failure for schedules with non-deterministic events
        let mut schedule3 = CashflowSchedule::new();
        schedule3.add_event(CashflowEvent::Deterministic(100.0));

        let mut schedule4 = CashflowSchedule::new();
        schedule4.add_event(CashflowEvent::Stochastic(
            StochasticCashflowEvent::Uniform { min: 10.0, max: 20.0 }
        ));

        assert!(schedule3.combine(&schedule4).is_none());
    }

    #[test]
    fn test_builder() {
        // Use the builder to construct a cashflow schedule.
        let schedule = CashflowScheduleBuilder::new()
            .add_event(CashflowEvent::Deterministic(100.0))
            .add_event(CashflowEvent::Stochastic(
                StochasticCashflowEvent::Uniform { min: 10.0, max: 20.0 }
            ))
            .build();

        assert_eq!(schedule.events.len(), 2);

        if let CashflowEvent::Deterministic(val) = schedule.events[0] {
            assert!((val - 100.0).abs() < f64::EPSILON);
        } else {
            panic!("Expected deterministic event");
        }
        if let CashflowEvent::Stochastic(StochasticCashflowEvent::Uniform { min, max }) = &schedule.events[1] {
            assert!((*min - 10.0).abs() < f64::EPSILON);
            assert!((*max - 20.0).abs() < f64::EPSILON);
        } else {
            panic!("Expected uniform stochastic event");
        }
    }
}

