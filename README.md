# Cashflow Analysis

Cashflow Analysis is a Rust crate that provides a flexible toolkit for defining, simulating, and analyzing sequential cashflow events. It is especially useful for financial planning, risk assessment, and investment analysis by allowing both deterministic and several stochastic models (e.g., Normal, Log-Normal, Uniform, Exponential) to model uncertainty and variability.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Defining Cashflow Events](#defining-cashflow-events)
  - [Building a Cashflow Schedule](#building-a-cashflow-schedule)
  - [Running a Monte Carlo Simulation](#running-a-monte-carlo-simulation)
- [Examples](#examples)
- [Limitations & Future Work](#limitations--future-work)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Flexible Event Definition:**  
  Define cashflow events using the `CashflowEvent` enum, which supports both deterministic events and events drawn from stochastic models.

- **Sequential Scheduling:**  
  Organize your events into a `CashflowSchedule` where each event represents a specific time period (e.g., quarterly), allowing for cumulative cashflow analysis.

- **Monte Carlo Simulation:**  
  Run repeated simulation trials to generate statistical summaries such as the mean, variance, and uncertainty intervals (95% confidence intervals) for each time period.

- **Composable Transformations:**  
  Adjust your cashflow values with operations like scaling (for inflation adjustments) and merge multiple cashflow schedules using methods like appending or combining schedules.

## Installation

To use this crate, add the following dependency to your `Cargo.toml`:

```toml
[dependencies]
cashflow-analysis = "0.1.0"
```

Additionally, ensure you have Rust and Cargo installed. This crate depends on [rand](https://crates.io/crates/rand), [rand_distr](https://crates.io/crates/rand_distr), and [rayon](https://crates.io/crates/rayon) for random number generation and parallel processing.

## Usage

### Defining Cashflow Events

Events are defined using the `CashflowEvent` enum. For example, you can create deterministic events or events that follow a statistical distribution:

```rust
use cashflow_analysis::CashflowEvent;

let event1 = CashflowEvent::Deterministic(100.0); // A fixed cashflow of 100.
let event2 = CashflowEvent::Uniform { min: 10.0, max: 20.0 }; // A stochastic event uniformly drawn from [10, 20].
```

### Building a Cashflow Schedule

A cashflow schedule represents a series of events (e.g., by quarter), allowing cumulative analysis:

```rust
use cashflow_analysis::{CashflowSchedule, CashflowEvent};

let mut schedule = CashflowSchedule::new();
schedule.add_event(CashflowEvent::Deterministic(100.0));
schedule.add_event(CashflowEvent::Uniform { min: 10.0, max: 20.0 });
```

### Running a Monte Carlo Simulation

The crate's Monte Carlo simulation functionality lets you perform repeated trials of your schedule:

```rust
use cashflow_analysis::{CashflowSchedule, CashflowEvent};
use rand::thread_rng;

fn main() {
    let mut schedule = CashflowSchedule::new();
    schedule.add_event(CashflowEvent::Deterministic(100.0));
    schedule.add_event(CashflowEvent::Uniform { min: 10.0, max: 20.0 });
    
    // Use the default Monte Carlo simulation with 10,000 trials.
    let result = schedule.run_monte_carlo(10_000, thread_rng()).unwrap();
    
    // Print the simulation statistics per quarter.
    for stat in result.quarter_stats {
        println!(
            "Quarter {}: Mean: {:.2}, Variance: {:.2}, Min: {:.2}, Max: {:.2}, CI: [{:.2}, {:.2}]",
            stat.quarter, stat.mean, stat.variance, stat.min, stat.max, stat.lower_bound, stat.upper_bound
        );
    }
}
```

## Examples

- **Financial Planning and Forecasting:**  
  Model future cashflows with variability to better plan for liquidity needs.

- **Risk Assessment:**  
  Use the simulation results to understand potential variability and extreme outcomes in investment returns or project cashflows.

- **Investment Analysis:**  
  Compare deterministic versus stochastic cashflow scenarios and compute confidence intervals for key performance metrics.

For more detailed samples, please check the examples directory in the repository if available.

## Limitations & Future Work

- **Additional Models:**  
  Support for additional probability distributions and advanced financial models is planned.

- **Enhanced APIs:**  
  We envision more intuitive APIs for composing and transforming complex cashflow events.

## Contributing

Contributions are very welcome! Please open issues or submit pull requests on the [GitHub repository](https://github.com/SamuelSchlesinger/cashflow-analysis). Follow the existing code style and include tests where applicable.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Thank you for exploring the **Cashflow Analysis** crate. Happy simulating!
