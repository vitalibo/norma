# Norma Overview

Use this page to understand the core building blocks of Norma and jump to the right guide for deeper details.

## Norma in Brief

Norma is a resilient data validation framework for Pandas and PySpark DataFrames. It ingests JSON Schema documents or
Python-based definitions, validates incoming datasets, and records any rule failures alongside the original data instead
of halting your pipeline. Start with the [project README](../readme.md) for installation instructions and a quick tour
of end-to-end validation flows.

## Key Guides

- [Engine Support](engine.md) – compare the Pandas and PySpark adapters, their error reporting styles, and extension
  points for custom engines.
- [Rules Reference](reference.md) – look up every built-in rule, coercion behaviour, and engine-specific type mapping.
- Python API
    - [Schema class](api/schema.md) - create and validate schemas.
    - [Column class](api/column.md) - define columns and their rules.
    - [Rules](api/rules.md) - apply built-in and custom rules to columns.
