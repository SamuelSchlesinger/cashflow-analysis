# Cashflow Analysis

Cashflow Analysis is a Rust crate that provides a flexible toolkit for defining, simulating, and analyzing sequential cashflow events. It is especially useful for financial planning, risk assessment, and investment analysis. The crate supports both deterministic events and a wide range of stochastic models—such as Normal, Log-Normal, Uniform, Exponential distributions—as well as composite and scaled events to model real-world uncertainty and variability.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Defining Cashflow Events](#defining-cashflow-events)
  - [Building a Cashflow Schedule](#building-a-cashflow-schedule)
  - [Running a Monte Carlo Simulation](#running-a-monte-carlo-simulation)
  - [Advanced Usage](#advanced-usage)
- [Examples](#examples)
- [Limitations & Future Work](#limitations--future-work)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Flexible Event Definition:**  
  Define cashflow events using the `CashflowEvent` enum. Supported variants include:
  - `Deterministic(f64)`: A fixed cashflow value.
  - `Normal { mean, std_dev }`: A payout drawn from a Normal distribution.
  - `LogNormal { location, scale }`: A payout drawn from a LogNormal distribution.
  - `Uniform { min, max }`: A payout drawn uniformly from the specified range.
  - `Exponential { rate }`: A payout drawn from an Exponential distribution.
  - `Discrete { outcomes }`: A weighted discrete event, where each outcome is another `CashflowEvent`.
  - `Composite { events }`: A composite event that aggregates the outcomes of multiple events.
  - `Scaled { event, factor }`: An event whose outcome is the result of an inner event scaled by a constant factor.

- **Sequential Scheduling:**  
  Organize your events into a `CashflowSchedule`, where each event represents a specific time period (e.g., a quarter). The schedule naturally accumulates cashflows over time.

- **Monte Carlo Simulation:**  
  Run repeated simulation trials (Monte Carlo simulations) to generate comprehensive statistical summaries—including mean, variance, and uncertainty intervals (95% confidence intervals) for each period.

- **Composable Transformations:**  
  Easily manipulate your cashflow schedules:
  - **Scaling:** Adjust cashflow values (e.g., for inflation) via the `.scale()` method.
  - **Merging Schedules:** Combine schedules with methods like appending (`append_schedule`) or element-wise addition using [`combine`].
  - **Builder Pattern:** Construct complex schedules using the fluent builder interface (`CashflowSchedule::builder()`).

## Installation

Add the following dependency to your `Cargo.toml`:

```toml
[dependencies]
cashflow-analysis = "0.1.0"
```

Ensure you have Rust and Cargo installed. This crate leverages the following libraries:
- [rand](https://crates.io/crates/rand)
- [rand_distr](https://crates.io/crates/rand_distr)
- [rayon](https://crates.io/crates/rayon)

for random number generation, statistical distributions, and parallel processing, respectively.

## Usage

### Defining Cashflow Events

The `CashflowEvent` enum allows you to define various types of cashflow events. For instance:

```rust
use cashflow_analysis::CashflowEvent;

// A deterministic cashflow of 100.
let event1 = CashflowEvent::Deterministic(100.0);

// A stochastic event with value uniformly drawn from [10, 20].
let event2 = CashflowEvent::Uniform { min: 10.0, max: 20.0 };

// A Normal distribution event.
let event3 = CashflowEvent::Normal { mean: 100.0, std_dev: 15.0 };

// A composite event combining multiple events.
let event4 = CashflowEvent::Composite {
    events: vec![
        CashflowEvent::Deterministic(50.0),
        cashflow_analysis::CashflowEvent::Uniform { min: 5.0, max: 15.0 },
    ],
};

// A scaled event (scaling a deterministic event by 1.1)
let event5 = CashflowEvent::Scaled {
    event: Box::new(CashflowEvent::Deterministic(200.0)),
    factor: 1.1,
};
```

### Building a Cashflow Schedule

The `CashflowSchedule` collects events sequentially (typically by quarter). You can build a schedule by adding events individually or using the builder:

```rust
use cashflow_analysis::{CashflowSchedule, CashflowEvent};

// Using the new() method:
let mut schedule = CashflowSchedule::new();
schedule.add_event(CashflowEvent::Deterministic(100.0));
schedule.add_event(CashflowEvent::Uniform { min: 10.0, max: 20.0 });

// Alternatively, using the builder pattern:
let schedule = CashflowSchedule::builder()
    .add_event(CashflowEvent::Deterministic(100.0))
    .add_event(CashflowEvent::Uniform { min: 10.0, max: 20.0 })
    .build();
```

### Running a Monte Carlo Simulation

The crate performs Monte Carlo simulations to compute cumulative statistics across time periods. For example:

```rust
use cashflow_analysis::{CashflowSchedule, CashflowEvent};
use rand::thread_rng;

fn main() {
    let mut schedule = CashflowSchedule::new();
    schedule.add_event(CashflowEvent::Deterministic(100.0));
    schedule.add_event(CashflowEvent::Uniform { min: 10.0, max: 20.0 });
    
    // Run a simulation with 10,000 trials using the default configuration.
    let result = schedule.run_monte_carlo(10_000, thread_rng()).unwrap();
    
    // Print the statistics for each period.
    for stat in result.quarter_stats {
        println!(
            "Quarter {}: Mean: {:.2}, Variance: {:.2}, Min: {:.2}, Max: {:.2}, CI: [{:.2}, {:.2}]",
            stat.quarter, stat.mean, stat.variance, stat.min, stat.max, stat.lower_bound, stat.upper_bound
        );
    }
}
```

### Advanced Usage

- **Scaling for Adjustments:** Use the `.scale()` method to adjust cashflows (for instance, for inflation).
- **Merging Schedules:** Combine schedules using `append_schedule` or perform element-wise addition with `combine`.
- **Custom Simulation Configuration:** For more control over simulation percentile thresholds, use `run_monte_carlo_with_config(...)`.

## Examples

Beyond the examples above, you can find further guidance and sample code in the [examples](./examples) directory of the GitHub repository.

## Limitations & Future Work

- **Additional Models:**  
  Future versions will consider support for more probability distributions and advanced financial models.
- **Enhanced APIs:**  
  We plan to introduce more intuitive APIs for composing and transforming complex cashflow events, as well as robust error handling mechanisms.
- **Performance & Flexibility:**  
  Further improvements may include more configuration options for Monte Carlo simulations and performance optimizations.

## Contributing

Contributions are very welcome! Please open issues or submit pull requests on the [GitHub repository](https://github.com/SamuelSchlesinger/cashflow-analysis). Follow the existing code style and include tests where applicable.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Thank you for exploring the **Cashflow Analysis** crate. Happy simulating!
