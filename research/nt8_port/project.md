# Project NT8 Native Port

## Define
Port the reference Python decider to C# Native NinjaScript, avoiding the bridge pattern for safety and performance (Architecture B).

## Measure
Bar-by-bar parity with >= 99% agreement against a 20-day golden reference set output by Python.

## Analyze
- Emitting golden vectors per 1m bar.
- Creating the Entry Port (reduced combiner streams).

## Improve
- Implement quantile matched thresholds in C#.
- Port logic cleanly.

## Control
Version the `-RC` scripts and ensure review before merging into `Strategies/`.
