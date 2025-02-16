//! # Cashflow Analysis
//!
//! The `cashflow-analysis` crate provides a comprehensive toolkit for defining, simulating, and analyzing
//! sequential cashflow events over time (typically by quarter). It supports both deterministic cashflows and
//! various stochastic models (e.g. Normal, Log-Normal, Uniform, Exponential), enabling you to model uncertainty
//! and variability. This makes it especially useful for financial planning, risk assessment, and investment analysis.
//!
//! ## Features
//!
//! - **Flexible Event Definition:** Define cashflow events using the [`CashflowEvent`] enum.
//! - **Sequential Scheduling:** Compose events into a [`CashflowSchedule`] where each event represents a quarter.
//! - **Monte Carlo Simulation:** Execute repeated random simulations (Monte Carlo) to aggregate and analyze
//!   cumulative cashflow distributions across time. The results include practical statistical summaries such as the
//!   mean, variance, and uncertainty intervals (95% confidence intervals), which help in assessing risk.
//! - **Composable Transformations:** Easily adjust and compose cashflow schedules using operations like scaling
//!   (for factors such as inflation adjustments) and methods to append or combine schedules.
//!
//! ## Examples
//!
//! Simulate a simple cashflow schedule:
//!
//! ```rust
//! use cashflow_analysis::{CashflowSchedule, CashflowEvent};
//! use rand::thread_rng;
//!
//! let mut schedule = CashflowSchedule::new();
//! schedule.add_event(CashflowEvent::Deterministic(100.0));
//! schedule.add_event(CashflowEvent::Uniform { min: 10.0, max: 20.0 });
//!
//! let result = schedule.run_monte_carlo(10_000, thread_rng()).unwrap();
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

use rand::distr::{weighted::WeightedIndex, Distribution, Uniform};
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Exp, LogNormal, Normal};
use rayon::prelude::*;

/// Custom error type for simulation errors.
#[derive(Debug)]
pub enum SimulationError {
    InvalidParameter(String),
    // Extend with other error types as needed.
}

/// Trait to abstract simulation behavior.
pub trait Simulate {
    fn simulate(&self, rng: &mut impl Rng) -> Result<f64, SimulationError>;
}

/// Represents a cashflow event. This unified and recursive type enables you to model a wide variety
/// of cashflow scenarios. It supports:
///
/// - **Single-payout events:** Such as a fixed (deterministic) cashflow or a cashflow drawn from a stochastic model.
/// - **Recursive events:** Where outcomes of one event may themselves be cashflow events (as with discrete events),
///   or a group of events is combined (composite events).
///
/// This flexibility lets you capture complex real-world cashflow patterns within a simple API.
///
/// # Variants
///
/// - `Deterministic(f64)`: A fixed cashflow value.
/// - `Normal { mean, std_dev }`: A payout drawn from a Normal distribution.
/// - `LogNormal { location, scale }`: A payout drawn from a LogNormal distribution.
/// - `Uniform { min, max }`: A payout drawn uniformly from [min, max].
/// - `Exponential { rate }`: A payout drawn from an Exponential distribution.
/// - `Discrete { outcomes }`: A discrete weighted event; each outcome is itself a [`CashflowEvent`] paired
///   with a probability.
/// - `Composite { events }`: A compound event that aggregates multiple events (the resulting payout is
///   the sum of the simulated outcomes from each sub-event).
/// - `Scaled { event: Box<CashflowEvent>, factor: f64 }`: **Scaled event variant:** Wraps an inner event and scales its simulated result by the given factor.
///
/// When simulated, the output of the inner event is multiplied by `factor`.
#[derive(Debug, Clone)]
pub enum CashflowEvent {
    /// A fixed cashflow value.
    Deterministic(f64),
    /// A payout drawn from a Normal distribution.
    Normal { mean: f64, std_dev: f64 },
    /// A payout drawn from a LogNormal distribution.
    LogNormal { location: f64, scale: f64 },
    /// A payout drawn uniformly from [min, max].
    Uniform { min: f64, max: f64 },
    /// A payout drawn from an Exponential distribution.
    Exponential { rate: f64 },
    /// A discrete event where each outcome is a [`CashflowEvent`] paired with a probability.
    Discrete { outcomes: Vec<(CashflowEvent, f64)> },
    /// A compound event that aggregates multiple events.
    Composite { events: Vec<CashflowEvent> },
    /// **Scaled event variant:** Wraps an inner event and scales its simulated result by the given factor.
    ///
    /// When simulated, the output of the inner event is multiplied by `factor`.
    Scaled {
        event: Box<CashflowEvent>,
        factor: f64,
    },
}

/// A schedule of cashflow events occurring sequentially by quarter.
///
/// Each event in the schedule represents the cashflow for a specific period (by default, a quarter).
/// The sequential order is important since cashflows are accumulated over time.
///
/// Use helper methods like [`CashflowEvent::scale`], [`CashflowSchedule::append_schedule`], and [`CashflowSchedule::combine`] to transform or compose schedules.
/// These operations allow you to adjust your cashflow values (e.g. to account for inflation) or merge cashflow streams.
#[derive(Debug, Clone)]
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
        CashflowSchedule { events: Vec::new() }
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
    /// This applies the [`CashflowEvent::scale`] method of [`CashflowEvent`] to each event.
    pub fn scale(&self, factor: f64) -> CashflowSchedule {
        let events = self.events.iter().map(|ev| ev.scale(factor)).collect();
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
        let combined_events = self
            .events
            .iter()
            .zip(other.events.iter())
            .map(|(a, b)| match (a, b) {
                (CashflowEvent::Deterministic(x), CashflowEvent::Deterministic(y)) => {
                    Some(CashflowEvent::Deterministic(x + y))
                }
                _ => return None,
            })
            .collect::<Option<Vec<_>>>()?;
        Some(CashflowSchedule::with_events(combined_events))
    }

    /// Executes a Monte Carlo simulation with custom configuration.
    pub fn run_monte_carlo_with_config(
        &self,
        num_trials: usize,
        rng: impl Rng,
        config: &SimulationConfig,
    ) -> Result<SimulationResult, SimulationError> {
        let num_quarters = self.num_quarters();

        // Generate a seed for each trial to initialize a thread‐local RNG.
        let mut seed_rng = rng;
        let trial_seeds: Vec<u64> = (0..num_trials).map(|_| seed_rng.random::<u64>()).collect();

        // Run simulation trials in parallel.
        let trial_results: Result<Vec<Vec<f64>>, SimulationError> = trial_seeds
            .into_par_iter()
            .map(|seed| {
                // Create a new thread-local RNG for each trial.
                let mut local_rng = rand::rngs::StdRng::seed_from_u64(seed);
                let mut cumulative = 0.0;
                let mut trial_outcomes = Vec::with_capacity(num_quarters);
                for event in &self.events {
                    let value = simulate_cashflow_event(event, &mut local_rng)?;
                    cumulative += value;
                    trial_outcomes.push(cumulative);
                }
                Ok(trial_outcomes)
            })
            .collect();
        let outcomes_per_trial = trial_results?;

        // Restructure outcomes so that outcomes for a given quarter are grouped together.
        let mut outcomes: Vec<Vec<f64>> = vec![Vec::with_capacity(num_trials); num_quarters];
        for trial in outcomes_per_trial {
            for (q, value) in trial.into_iter().enumerate() {
                outcomes[q].push(value);
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
                let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                // Compute your custom confidence intervals based on the config percentiles.
                let mut sorted = values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let lower_index =
                    (((n as f64) * config.lower_percentile).round() as usize).min(n - 1);
                let upper_index =
                    (((n as f64) * config.upper_percentile).round() as usize).min(n - 1);
                let lower_bound = sorted[lower_index];
                let upper_bound = sorted[upper_index];

                QuarterStatistics {
                    quarter: q + 1,
                    mean,
                    variance,
                    min: min_val,
                    max: max_val,
                    lower_bound,
                    upper_bound,
                }
            })
            .collect();

        Ok(SimulationResult { quarter_stats })
    }

    /// Executes a Monte Carlo simulation with default configuration.
    pub fn run_monte_carlo(
        &self,
        num_trials: usize,
        rng: impl Rng,
    ) -> Result<SimulationResult, SimulationError> {
        self.run_monte_carlo_with_config(num_trials, rng, &SimulationConfig::default())
    }

    pub fn builder() -> CashflowScheduleBuilder {
        CashflowScheduleBuilder::new()
    }
}

/// A builder for constructing a CashflowSchedule.
pub struct CashflowScheduleBuilder {
    events: Vec<CashflowEvent>,
}

impl CashflowScheduleBuilder {
    pub fn new() -> Self {
        CashflowScheduleBuilder { events: Vec::new() }
    }

    pub fn add_event(mut self, event: CashflowEvent) -> Self {
        self.events.push(event);
        self
    }

    pub fn build(self) -> CashflowSchedule {
        CashflowSchedule {
            events: self.events,
        }
    }
}

// --------------------------------------------------------------------
// Helper function to simulate a single cashflow event
fn simulate_cashflow_event<R: Rng + ?Sized>(
    event: &CashflowEvent,
    rng: &mut R,
) -> Result<f64, SimulationError> {
    match event {
        CashflowEvent::Deterministic(value) => Ok(*value),
        CashflowEvent::Normal { mean, std_dev } => {
            let normal = Normal::new(*mean, *std_dev)
                .map_err(|e| SimulationError::InvalidParameter(e.to_string()))?;
            Ok(normal.sample(rng))
        }
        CashflowEvent::LogNormal { location, scale } => {
            let lognormal = LogNormal::new(*location, *scale)
                .map_err(|e| SimulationError::InvalidParameter(e.to_string()))?;
            Ok(lognormal.sample(rng))
        }
        CashflowEvent::Uniform { min, max } => {
            let uniform = Uniform::new_inclusive(*min, *max)
                .map_err(|e| SimulationError::InvalidParameter(e.to_string()))?;
            Ok(uniform.sample(rng))
        }
        CashflowEvent::Exponential { rate } => {
            let exp =
                Exp::new(*rate).map_err(|e| SimulationError::InvalidParameter(e.to_string()))?;
            Ok(exp.sample(rng))
        }
        CashflowEvent::Discrete { outcomes } => {
            let weights: Vec<f64> = outcomes.iter().map(|&(_, prob)| prob).collect();
            let dist = WeightedIndex::new(&weights)
                .map_err(|e| SimulationError::InvalidParameter(e.to_string()))?;
            let index = dist.sample(rng);
            let (ref outcome, _) = outcomes[index];
            simulate_cashflow_event(outcome, rng)
        }
        CashflowEvent::Composite { events } => {
            let mut sum = 0.0;
            for ev in events {
                sum += simulate_cashflow_event(ev, rng)?;
            }
            Ok(sum)
        }
        CashflowEvent::Scaled { event, factor } => {
            Ok(simulate_cashflow_event(event, rng)? * factor)
        }
    }
}

// --------------------------------------------------------------------
// Structs to store simulation statistics
/// Statistics for a given quarter of the simulation.
///
/// In addition to the basic summary statistics, this structure now provides uncertainty
/// bounds computed as the 2.5th and 97.5th percentiles (approximating a 95% confidence interval).
#[derive(Debug)]
pub struct QuarterStatistics {
    /// Quarter number (1-indexed)
    pub quarter: usize,
    /// Sample mean of the cumulative cashflow for this quarter.
    pub mean: f64,
    /// Sample variance (using n-1 as the denominator).
    pub variance: f64,
    /// Minimum cumulative cashflow observed.
    pub min: f64,
    /// Maximum cumulative cashflow observed.
    pub max: f64,
    /// Lower bound (2.5th percentile) of the cumulative cashflow.
    pub lower_bound: f64,
    /// Upper bound (97.5th percentile) of the cumulative cashflow.
    pub upper_bound: f64,
}

/// Simulation results, containing per-quarter statistics.
#[derive(Debug)]
pub struct SimulationResult {
    pub quarter_stats: Vec<QuarterStatistics>,
}

impl CashflowEvent {
    /// Returns a new scaled cashflow event.
    ///
    /// When simulated, the output of the underlying event is multiplied by the given `factor`.
    /// For linear events, this scales the parameters accordingly. For instance, a Normal event's mean and standard deviation
    /// are multiplied by `factor`, while an Exponential event remains unchanged.
    pub fn scale(&self, factor: f64) -> CashflowEvent {
        match self {
            CashflowEvent::Deterministic(val) => CashflowEvent::Deterministic(val * factor),
            CashflowEvent::Normal { mean, std_dev } => CashflowEvent::Normal {
                mean: mean * factor,
                std_dev: std_dev * factor,
            },
            CashflowEvent::LogNormal { location, scale } => CashflowEvent::LogNormal {
                location: location * factor,
                scale: scale * factor,
            },
            CashflowEvent::Uniform { min, max } => CashflowEvent::Uniform {
                min: min * factor,
                max: max * factor,
            },
            CashflowEvent::Exponential { rate } => CashflowEvent::Exponential { rate: *rate },
            CashflowEvent::Discrete { outcomes } => {
                let out = outcomes
                    .iter()
                    .map(|(ev, prob)| (ev.scale(factor), *prob))
                    .collect();
                CashflowEvent::Discrete { outcomes: out }
            }
            CashflowEvent::Composite { events } => {
                let evs = events.iter().map(|ev| ev.scale(factor)).collect();
                CashflowEvent::Composite { events: evs }
            }
            CashflowEvent::Scaled {
                event,
                factor: inner_factor,
            } => {
                // Combine the inner scaling factor with the new one.
                event.scale(inner_factor * factor)
            }
        }
    }
}

// Implement Simulate for CashflowEvent.
impl Simulate for CashflowEvent {
    fn simulate(&self, rng: &mut impl Rng) -> Result<f64, SimulationError> {
        simulate_cashflow_event(self, rng)
    }
}

/// Struct to configure simulation parameters such as confidence interval percentiles.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    pub lower_percentile: f64,
    pub upper_percentile: f64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        SimulationConfig {
            lower_percentile: 0.025,
            upper_percentile: 0.975,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_deterministic_simulation() {
        let mut schedule = CashflowSchedule::new();
        schedule.add_event(CashflowEvent::Deterministic(100.0));
        schedule.add_event(CashflowEvent::Deterministic(50.0));

        let rng = StdRng::seed_from_u64(42);
        let result = schedule.run_monte_carlo(100, rng).unwrap();

        // We expect each trial to yield [100.0, 150.0].
        assert_eq!(result.quarter_stats.len(), 2);

        let stat_q1 = &result.quarter_stats[0];
        assert!(
            (stat_q1.mean - 100.0).abs() < f64::EPSILON,
            "Expected Q1 mean 100.0, got {}",
            stat_q1.mean
        );
        assert!((stat_q1.variance - 0.0).abs() < f64::EPSILON);
        assert!((stat_q1.min - 100.0).abs() < f64::EPSILON);
        assert!((stat_q1.max - 100.0).abs() < f64::EPSILON);

        let stat_q2 = &result.quarter_stats[1];
        assert!(
            (stat_q2.mean - 150.0).abs() < f64::EPSILON,
            "Expected Q2 mean 150.0, got {}",
            stat_q2.mean
        );
        assert!((stat_q2.variance - 0.0).abs() < f64::EPSILON);
        assert!((stat_q2.min - 150.0).abs() < f64::EPSILON);
        assert!((stat_q2.max - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_schedule_simulation() {
        let schedule = CashflowSchedule::new();
        let rng = StdRng::seed_from_u64(123);
        let result = schedule.run_monte_carlo(100, rng).unwrap();
        // With no events, the simulation result should contain no quarter statistics.
        assert_eq!(result.quarter_stats.len(), 0);
    }

    #[test]
    fn test_uniform_simulation_range() {
        // Create a schedule with a uniform random event for Q1,
        // followed by a deterministic event for Q2.
        let mut schedule = CashflowSchedule::new();
        schedule.add_event(CashflowEvent::Uniform {
            min: 10.0,
            max: 20.0,
        });
        schedule.add_event(CashflowEvent::Deterministic(30.0));

        let rng = StdRng::seed_from_u64(789);
        let result = schedule.run_monte_carlo(10_000, rng).unwrap();

        // For Q1, the simulated values should lie in the range [10, 20].
        let stat_q1 = &result.quarter_stats[0];
        assert!(
            stat_q1.mean > 12.0 && stat_q1.mean < 18.0,
            "Uniform event mean expected between 12 and 18, got {}",
            stat_q1.mean
        );
        assert!(
            stat_q1.min >= 10.0 && stat_q1.max <= 20.0,
            "Uniform event range expected between [10,20], got [{}, {}]",
            stat_q1.min,
            stat_q1.max
        );

        // For Q2, the cumulative value is the Q1 value plus a deterministic 30.
        let stat_q2 = &result.quarter_stats[1];
        assert!((stat_q2.mean - (stat_q1.mean + 30.0)).abs() < 1e-6);
        assert!((stat_q2.min - (stat_q1.min + 30.0)).abs() < 1e-6);
        assert!((stat_q2.max - (stat_q1.max + 30.0)).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_simulation() {
        // Create a schedule with an exponential event (rate = 0.1, theoretical mean = 10).
        let mut schedule = CashflowSchedule::new();
        schedule.add_event(CashflowEvent::Exponential { rate: 0.1 });

        let rng = StdRng::seed_from_u64(2021);
        let result = schedule.run_monte_carlo(10_000, rng).unwrap();

        let stat_q1 = &result.quarter_stats[0];
        // Ensure the mean is close to 10.
        assert!(
            (stat_q1.mean - 10.0).abs() < 1.0,
            "Exponential event mean not as expected: {}",
            stat_q1.mean
        );
    }

    #[test]
    fn test_scaling() {
        // Create a schedule with one deterministic and one uniform event.
        let mut schedule = CashflowSchedule::new();
        schedule.add_event(CashflowEvent::Deterministic(100.0));
        schedule.add_event(CashflowEvent::Uniform {
            min: 10.0,
            max: 20.0,
        });

        // Test scaling
        let scaled_schedule = schedule.scale(2.0);
        // First event: 100.0 becomes 200.0
        if let CashflowEvent::Deterministic(val) = scaled_schedule.events[0] {
            assert!(
                (val - 200.0).abs() < f64::EPSILON,
                "Expected 200.0, got {}",
                val
            );
        } else {
            panic!("Expected a Deterministic event");
        }
        // Second event: Uniform (10,20) becomes Uniform (20,40)
        if let CashflowEvent::Uniform { min, max } = &scaled_schedule.events[1] {
            assert!(
                (*min - 20.0).abs() < f64::EPSILON,
                "Expected scaled min 20, got {}",
                min
            );
            assert!(
                (*max - 40.0).abs() < f64::EPSILON,
                "Expected scaled max 40, got {}",
                max
            );
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

        let combined = schedule1.combine(&schedule2).unwrap();
        assert_eq!(combined.events.len(), 2);

        if let CashflowEvent::Deterministic(val) = combined.events[0] {
            assert!(
                (val - 110.0).abs() < f64::EPSILON,
                "Expected 110.0, got {}",
                val
            );
        } else {
            panic!("Expected a deterministic event");
        }
        if let CashflowEvent::Deterministic(val) = combined.events[1] {
            assert!(
                (val - 220.0).abs() < f64::EPSILON,
                "Expected 220.0, got {}",
                val
            );
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
        schedule4.add_event(CashflowEvent::Uniform {
            min: 10.0,
            max: 20.0,
        });

        assert!(schedule3.combine(&schedule4).is_none());
    }

    #[test]
    fn test_run_monte_carlo() {
        let mut schedule = CashflowSchedule::new();
        schedule.add_event(CashflowEvent::Deterministic(100.0));
        schedule.add_event(CashflowEvent::Uniform {
            min: 10.0,
            max: 20.0,
        });
        let rng = StdRng::seed_from_u64(123);
        let result = schedule.run_monte_carlo(100, rng).unwrap();
        // Assert that the simulation produced some quarter statistics.
        assert!(
            !result.quarter_stats.is_empty(),
            "Expected non-empty quarter statistics"
        );
    }

    #[test]
    fn test_composite_event() {
        // Composite event: sum of a deterministic event and a uniform event.
        let composite_event = CashflowEvent::Composite {
            events: vec![
                CashflowEvent::Deterministic(50.0),
                CashflowEvent::Uniform {
                    min: 10.0,
                    max: 20.0,
                },
            ],
        };
        let mut schedule = CashflowSchedule::new();
        schedule.add_event(composite_event);
        let rng = StdRng::seed_from_u64(999);
        let result = schedule.run_monte_carlo(1_000, rng).unwrap();
        let stat = &result.quarter_stats[0];
        // Check that the composite outcome is roughly in the expected range (i.e. [60,70]).
        assert!(
            stat.mean >= 60.0 && stat.mean <= 70.0,
            "Composite event mean out of range: {}",
            stat.mean
        );
        // Also verify that the uncertainty bounds make sense.
        assert!(
            stat.lower_bound <= stat.mean && stat.mean <= stat.upper_bound,
            "Composite event uncertainty bounds incorrect: lower: {}, upper: {}, mean: {}",
            stat.lower_bound,
            stat.upper_bound,
            stat.mean
        );
    }

    #[test]
    fn test_discrete_event() {
        // Discrete event: two outcomes, 30.0 with probability 0.7 and 50.0 with probability 0.3.
        let discrete_event = CashflowEvent::Discrete {
            outcomes: vec![
                (CashflowEvent::Deterministic(30.0), 0.7),
                (CashflowEvent::Deterministic(50.0), 0.3),
            ],
        };
        let mut schedule = CashflowSchedule::new();
        schedule.add_event(discrete_event);
        let rng = StdRng::seed_from_u64(555);
        let result = schedule.run_monte_carlo(1_000, rng).unwrap();
        let stat = &result.quarter_stats[0];
        // Expected weighted mean: 0.7 * 30 + 0.3 * 50 = 34.0. Allow a tolerance.
        assert!(
            (stat.mean - 34.0).abs() < 5.0,
            "Discrete event mean expected near 34.0, got {}",
            stat.mean
        );
        // Check that the uncertainty bounds are sensible.
        assert!(
            stat.lower_bound <= stat.mean && stat.mean <= stat.upper_bound,
            "Discrete event uncertainty bounds incorrect: lower: {}, mean: {}, upper: {}",
            stat.lower_bound,
            stat.mean,
            stat.upper_bound
        );
    }

    #[test]
    fn test_uncertainty_bounds() {
        // Create a schedule with a uniform event.
        let mut schedule = CashflowSchedule::new();
        schedule.add_event(CashflowEvent::Uniform {
            min: 10.0,
            max: 20.0,
        });
        let rng = StdRng::seed_from_u64(789);
        let result = schedule.run_monte_carlo(10_000, rng).unwrap();
        let stat = &result.quarter_stats[0];
        // Check that the uncertainty bounds (lower_bound and upper_bound) enclose the mean.
        assert!(
            stat.lower_bound <= stat.mean,
            "Lower bound exceeds mean: {} > {}",
            stat.lower_bound,
            stat.mean
        );
        assert!(
            stat.mean <= stat.upper_bound,
            "Mean exceeds upper bound: {} > {}",
            stat.mean,
            stat.upper_bound
        );
    }

    // --- New tests to cover remaining variants and 100% coverage ---

    #[test]
    fn test_normal_event_transformation() {
        let event = CashflowEvent::Normal {
            mean: 100.0,
            std_dev: 10.0,
        };
        let scaled = event.scale(2.0);
        if let CashflowEvent::Normal { mean, std_dev } = scaled {
            assert!((mean - 200.0).abs() < f64::EPSILON);
            assert!((std_dev - 20.0).abs() < f64::EPSILON);
        } else {
            panic!("Expected Normal variant");
        }
    }

    #[test]
    fn test_lognormal_event_transformation() {
        let event = CashflowEvent::LogNormal {
            location: 50.0,
            scale: 5.0,
        };
        let scaled = event.scale(2.0);
        if let CashflowEvent::LogNormal { location, scale } = scaled {
            assert!((location - 100.0).abs() < f64::EPSILON);
            assert!((scale - 10.0).abs() < f64::EPSILON);
        } else {
            panic!("Expected LogNormal variant");
        }
    }

    #[test]
    fn test_exponential_event_transformation() {
        let event = CashflowEvent::Exponential { rate: 0.1 };
        let scaled = event.scale(2.0);
        if let CashflowEvent::Exponential { rate } = scaled {
            // Exponential events remain unchanged on scaling in our design.
            assert!((rate - 0.1).abs() < f64::EPSILON);
        } else {
            panic!("Expected Exponential variant");
        }
    }

    #[test]
    fn test_discrete_and_composite_transformation() {
        // Build a discrete event with deterministic outcomes.
        let discrete_event = CashflowEvent::Discrete {
            outcomes: vec![
                (CashflowEvent::Deterministic(20.0), 0.5),
                (CashflowEvent::Deterministic(40.0), 0.5),
            ],
        };
        let scaled_discrete = discrete_event.scale(2.0);
        if let CashflowEvent::Discrete { outcomes } = scaled_discrete {
            match &outcomes[0].0 {
                CashflowEvent::Deterministic(val) => assert!((val - 40.0).abs() < f64::EPSILON),
                _ => panic!("Expected Deterministic outcome"),
            }
            match &outcomes[1].0 {
                CashflowEvent::Deterministic(val) => assert!((val - 80.0).abs() < f64::EPSILON),
                _ => panic!("Expected Deterministic outcome"),
            }
        } else {
            panic!("Expected Discrete variant");
        }

        // Build a composite event.
        let composite_event = CashflowEvent::Composite {
            events: vec![
                CashflowEvent::Deterministic(10.0),
                CashflowEvent::Uniform {
                    min: 5.0,
                    max: 15.0,
                },
            ],
        };
        let scaled_composite = composite_event.scale(3.0);
        if let CashflowEvent::Composite { events } = scaled_composite {
            match &events[0] {
                CashflowEvent::Deterministic(val) => assert!((val - 30.0).abs() < f64::EPSILON),
                _ => panic!("Expected Deterministic event"),
            }
            match &events[1] {
                CashflowEvent::Uniform { min, max } => {
                    assert!((*min - 15.0).abs() < f64::EPSILON);
                    assert!((*max - 45.0).abs() < f64::EPSILON);
                }
                _ => panic!("Expected Uniform event"),
            }
        } else {
            panic!("Expected Composite variant");
        }
    }

    // --- New Monte Carlo Simulation Tests ---

    /// Test that a schedule with purely deterministic events returns identical statistics.
    #[test]
    fn test_monte_carlo_deterministic() {
        // Schedule with two deterministic events.
        let mut schedule = CashflowSchedule::new();
        schedule.add_event(CashflowEvent::Deterministic(100.0));
        schedule.add_event(CashflowEvent::Deterministic(50.0));

        let rng = StdRng::seed_from_u64(42);
        let result = schedule.run_monte_carlo(100, rng).unwrap();
        assert_eq!(result.quarter_stats.len(), 2);

        // For each quarter, the outcome should be fixed.
        for (i, stat) in result.quarter_stats.iter().enumerate() {
            let expected = if i == 0 { 100.0 } else { 150.0 };
            assert!(
                (stat.mean - expected).abs() < f64::EPSILON,
                "Expected quarter {} mean to be {}, got {}",
                i + 1,
                expected,
                stat.mean
            );
            assert_eq!(
                stat.variance, 0.0,
                "Variance should be zero for deterministic events"
            );
            assert_eq!(stat.min, expected);
            assert_eq!(stat.max, expected);
            // With one unique value across all trials, the confidence interval equals the mean.
            assert_eq!(stat.lower_bound, expected);
            assert_eq!(stat.upper_bound, expected);
        }
    }

    /// Test that a schedule with a single uniform event yields statistics close to theoretical predictions.
    #[test]
    fn test_monte_carlo_uniform_statistics() {
        // Uniform event between 10 and 20.
        let mut schedule = CashflowSchedule::new();
        schedule.add_event(CashflowEvent::Uniform {
            min: 10.0,
            max: 20.0,
        });
        let num_trials = 10_000;
        let rng = StdRng::seed_from_u64(1001);
        let result = schedule.run_monte_carlo(num_trials, rng).unwrap();
        let stat = &result.quarter_stats[0];

        // Theoretical properties for Uniform(a=10, b=20):
        // Mean = (a+b)/2 = 15, Variance = (b-a)²/12 = 100/12 ≈ 8.333...
        let theoretical_mean = 15.0;
        let theoretical_variance = 100.0 / 12.0;

        let tolerance_mean = 0.5; // Allowable deviation for mean.
        let tolerance_variance = 1.0; // Allowable deviation for variance.

        assert!(
            (stat.mean - theoretical_mean).abs() < tolerance_mean,
            "Uniform event mean expected near {}, got {}",
            theoretical_mean,
            stat.mean
        );
        assert!(
            (stat.variance - theoretical_variance).abs() < tolerance_variance,
            "Uniform event variance expected near {}, got {}",
            theoretical_variance,
            stat.variance
        );

        // Check that uncertainty bounds are ordered properly.
        assert!(
            stat.lower_bound <= stat.mean && stat.mean <= stat.upper_bound,
            "Uniform event: expected lower_bound <= mean <= upper_bound"
        );
        // Basic sanity: the confidence interval should have nonzero width.
        assert!(
            stat.upper_bound - stat.lower_bound > 0.0,
            "Expected non-zero confidence interval width"
        );
    }

    /// Test that a schedule with a single normal event yields sample statistics near the theoretical values.
    #[test]
    fn test_monte_carlo_normal_statistics() {
        // Normal event with mean 50 and std deviation 5.0 (variance 25).
        let norm_mean = 50.0;
        let norm_std = 5.0;
        let mut schedule = CashflowSchedule::new();
        schedule.add_event(CashflowEvent::Normal {
            mean: norm_mean,
            std_dev: norm_std,
        });
        let num_trials = 10_000;
        let rng = StdRng::seed_from_u64(2002);
        let result = schedule.run_monte_carlo(num_trials, rng).unwrap();
        let stat = &result.quarter_stats[0];

        let tolerance_mean = 0.5;
        let tolerance_variance = 2.0;

        assert!(
            (stat.mean - norm_mean).abs() < tolerance_mean,
            "Normal event mean expected near {}, got {}",
            norm_mean,
            stat.mean
        );
        assert!(
            (stat.variance - 25.0).abs() < tolerance_variance,
            "Normal event variance expected near {}, got {}",
            25.0,
            stat.variance
        );
        assert!(
            stat.lower_bound <= stat.mean && stat.mean <= stat.upper_bound,
            "Normal event: bounds ordering violated"
        );
    }

    /// Test that with a single trial, the statistics are trivial.
    #[test]
    fn test_single_trial_simulation() {
        // A schedule with one uniform event.
        let mut schedule = CashflowSchedule::new();
        schedule.add_event(CashflowEvent::Uniform {
            min: 10.0,
            max: 20.0,
        });
        let rng = StdRng::seed_from_u64(3030);
        let result = schedule.run_monte_carlo(1, rng).unwrap();
        let stat = &result.quarter_stats[0];

        // With one trial, variance must be zero and the min, max, and confidence bounds equal the single outcome.
        assert_eq!(
            stat.variance, 0.0,
            "Variance for a single trial should be zero"
        );
        assert_eq!(stat.lower_bound, stat.mean);
        assert_eq!(stat.upper_bound, stat.mean);
    }

    /// Test that a cumulative schedule (combining events across quarters) produces reasonable results.
    #[test]
    fn test_cumulative_simulation() {
        // Define a schedule with two events:
        // Quarter 1: Deterministic value of 100.
        // Quarter 2: Uniform event in [10,20]. The cumulative sum is 100 + (Uniform value).
        let mut schedule = CashflowSchedule::new();
        schedule.add_event(CashflowEvent::Deterministic(100.0));
        schedule.add_event(CashflowEvent::Uniform {
            min: 10.0,
            max: 20.0,
        });

        let num_trials = 10_000;
        let rng = StdRng::seed_from_u64(4040);
        let result = schedule.run_monte_carlo(num_trials, rng).unwrap();
        assert_eq!(result.quarter_stats.len(), 2);

        // Quarter 1 should have mean exactly 100.
        let stat1 = &result.quarter_stats[0];
        assert!(
            (stat1.mean - 100.0).abs() < 0.1,
            "Quarter 1 deterministic result expected 100, got {}",
            stat1.mean
        );

        // Quarter 2: cumulative mean should be 100 + the mean of Uniform [10,20] i.e., 100 + 15 = 115.
        let stat2 = &result.quarter_stats[1];
        assert!(
            (stat2.mean - 115.0).abs() < 0.5,
            "Quarter 2 cumulative mean expected near 115, got {}",
            stat2.mean
        );
    }

    #[test]
    fn test_confidence_interval_coverage() -> Result<(), SimulationError> {
        // Use a uniform event so that the distribution is known.
        let mut schedule = CashflowSchedule::new();
        schedule.add_event(CashflowEvent::Uniform {
            min: 10.0,
            max: 20.0,
        });
        let num_trials = 10_000;
        let rng = rand::rngs::StdRng::seed_from_u64(5050);
        let result = schedule.run_monte_carlo(num_trials, rng).unwrap();
        let stat = &result.quarter_stats[0];

        // Instead of manually re-simulating, we check that approximately 95% of outcomes fall between
        // the simulation's reported [lower_bound, upper_bound].
        // (The simulation already sorts and computes these quantiles internally.)
        // (The variable 'count_within' from a previous version was removed since it is not needed.)
        // For simplicity, we re-simulate manually with a new RNG instance (it will produce a different set
        // but still follow the same Uniform distribution):
        let mut manual_rng = rand::rngs::StdRng::seed_from_u64(9999);
        let mut manual_outcomes = Vec::with_capacity(num_trials);
        for _ in 0..num_trials {
            let outcome = simulate_cashflow_event(
                &CashflowEvent::Uniform {
                    min: 10.0,
                    max: 20.0,
                },
                &mut manual_rng,
            )?;
            manual_outcomes.push(outcome);
        }
        manual_outcomes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let count_within = manual_outcomes
            .iter()
            .filter(|&&x| x >= stat.lower_bound && x <= stat.upper_bound)
            .count();
        let percentage = (count_within as f64) / (num_trials as f64);
        assert!(
            (percentage - 0.95).abs() < 0.03,
            "Expected about 95% within CI, got {:.2}%",
            percentage * 100.0
        );
        Ok(())
    }
}
