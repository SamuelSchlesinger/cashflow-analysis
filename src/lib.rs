use rand::Rng;
use rand::distr::{Distribution, weighted::WeightedIndex, Uniform};
use rand_distr::{Normal, LogNormal, Exp};

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

/// A schedule of cashflow events that occur quarterly
pub struct CashflowSchedule {
    /// Vector of cashflow events, where each event occurs in sequential quarters
    /// The first event occurs in quarter 1, second in quarter 2, etc.
    pub events: Vec<CashflowEvent>,
}

impl CashflowSchedule {
    /// Creates a new empty cashflow schedule
    pub fn new() -> Self {
        CashflowSchedule {
            events: Vec::new()
        }
    }

    /// Creates a new cashflow schedule with the given events
    pub fn with_events(events: Vec<CashflowEvent>) -> Self {
        CashflowSchedule { events }
    }

    /// Adds a cashflow event to the schedule
    pub fn add_event(&mut self, event: CashflowEvent) {
        self.events.push(event);
    }

    /// Returns the number of quarters in the schedule
    pub fn num_quarters(&self) -> usize {
        self.events.len()
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

// --------------------------------------------------------------------
// Extend CashflowSchedule with a Monte Carlo simulation method
impl CashflowSchedule {
    /// Runs a Monte Carlo simulation on the cashflow schedule over a given number of trials.
    /// For each trial, it computes the cumulative cashflow per quarter and then calculates
    /// the mean, variance, minimum, and maximum for each quarter across all trials.
    pub fn run_monte_carlo(&self, num_trials: usize, mut rng: impl Rng) -> SimulationResult {
        let num_quarters = self.num_quarters();
        // Prepare a vector to record cumulative cashflow outcomes per quarter across trials
        let mut outcomes: Vec<Vec<f64>> = vec![Vec::with_capacity(num_trials); num_quarters];
        // Run simulation trials
        for _ in 0..num_trials {
            let mut cumulative = 0.0;
            for (q, event) in self.events.iter().enumerate() {
                let value = simulate_cashflow_event(event, &mut rng);
                cumulative += value;
                // Each quarter's outcomes records the cumulative value so far
                outcomes[q].push(cumulative);
            }
        }

        // Compute statistics for each quarter
        let quarter_stats = outcomes
            .into_iter()
            .enumerate()
            .map(|(q, values)| {
                let n = values.len() as f64;
                let mean = values.iter().sum::<f64>() / n;
                let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
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
}

