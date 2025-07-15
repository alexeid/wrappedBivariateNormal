# Wrapped Bivariate Normal Distribution

A Java implementation of the Wrapped Bivariate Normal distribution on the torus [0, 2π) × [0, 2π).

## Overview

The Wrapped Bivariate Normal distribution is useful for modeling circular/angular data in two dimensions, such as:
- Wind directions at different altitudes
- Phase angles in signal processing
- Directional data in circular statistics
- Angular measurements with correlation

## Features

- **Probability Density Function (PDF)** computation with numerical stability
- **Random sampling** from the distribution
- **Circular statistics** (means and variances)
- **Log-PDF** computation for numerical stability
- **Correlation** support between angular components

## Installation

### Maven
```xml
<dependency>
    <groupId>wrappednormal</groupId>
    <artifactId>wrapped-bivariate-normal</artifactId>
    <version>1.0.0</version>
</dependency>
