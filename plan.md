# Calculus Package Plan

This document outlines the plan for the `calculus` package, providing calculus operations that work with ratmath's various real number representations (oracles, intervals, Cauchy sequences). The design emphasizes real-agnosticism and optional symbolic capabilities.

## Package Structure

```
packages/calculus/
├── src/
│   ├── index.js              # Main exports and registration
│   ├── ratmath-module.js     # VariableManager module integration
│   ├── limits.js             # Limit computation
│   ├── derivatives.js        # Differentiation (numeric & symbolic)
│   ├── integrals.js          # Integration (numeric & symbolic)
│   ├── series.js             # Series and sequences
│   ├── differential-eq.js    # Basic ODE solvers
│   ├── optimization.js       # Min/max finding
│   ├── real-interface.js     # Abstraction over real representations
│   └── symbolic.js           # Symbolic differentiation/integration
├── help/
│   ├── calculus.txt          # Main package help
│   ├── derivatives.txt       # Derivative help
│   ├── integrals.txt         # Integration help
│   └── series.txt            # Series help
├── tests/
│   ├── derivatives.test.js
│   ├── integrals.test.js
│   ├── limits.test.js
│   └── series.test.js
└── package.json
```

---

## Philosophy

### Real-Agnostic Design

The calculus package should work with multiple real number implementations:

1. **Oracles** (`@ratmath/oracles`): Real numbers as interval-refining functions
2. **Intervals** (`@ratmath/core`): Interval arithmetic with rational endpoints
3. **Cauchy sequences**: Computable reals as convergent sequences
4. **Floats**: Standard floating-point (for comparison/fallback)

All implementations should conform to a common **Real Interface**:

```javascript
interface Real {
  // Get interval containing the real with width ≤ epsilon
  narrow(epsilon: Rational): RationalInterval;
  
  // Compare with rational (returns -1, 0, 1, or undefined if undecidable)
  compare(q: Rational): number | undefined;
  
  // Arithmetic operations return new Reals
  add(other: Real): Real;
  mul(other: Real): Real;
  neg(): Real;
  inv(): Real;  // May fail if zero
  
  // Optional: symbolic derivative if known
  derivative?: () => Real;
}
```

### Symbolic Enhancement

Functions can carry optional symbolic metadata:

```
f := x -> x^2
Set(f, "symbolic", "x^2")
Set(f, "derivative", x -> 2*x)
Set(f, "integral", x -> x^3/3)
```

When available, algorithms use symbolic information; otherwise fall back to numeric.

---

## Category 1: Limits

### Basic Limits

| Function | Signature | Description |
|----------|-----------|-------------|
| `Limit` | `Limit(f, x, a, opts?)` | lim_{x→a} f(x) |
| `LimitLeft` | `LimitLeft(f, x, a)` | Left-hand limit |
| `LimitRight` | `LimitRight(f, x, a)` | Right-hand limit |
| `LimitInf` | `LimitInf(f, x, dir?)` | Limit as x → ±∞ |

### Limit Options

```
Limit(f, x, 0, {
  precision: 1/1000000,   # Target precision
  maxIter: 1000,          # Maximum iterations
  method: "sequence",     # "sequence", "richardson", "symbolic"
  showSteps: 0            # Show convergence steps
})
```

### Limit Computation Methods

1. **Sequence method**: Evaluate f at x_n → a, check convergence
2. **Richardson extrapolation**: Accelerate convergence
3. **Symbolic**: If derivative info available, use L'Hôpital's rule
4. **Interval**: Use interval arithmetic to bound limit

### Special Limits

| Function | Signature | Description |
|----------|-----------|-------------|
| `Continuous` | `Continuous(f, a)` | Check if f continuous at a |
| `Discontinuities` | `Discontinuities(f, a, b)` | Find discontinuities in [a,b] |
| `RemovableDisc` | `RemovableDisc(f, a)` | Check if discontinuity removable |

---

## Category 2: Derivatives

### Basic Differentiation

| Function | Signature | Description |
|----------|-----------|-------------|
| `Derivative` | `Derivative(f, x?, n?)` | nth derivative of f |
| `D` | `D(f)` | Shorthand for first derivative |
| `PartialD` | `PartialD(f, var)` | Partial derivative |
| `Gradient` | `Gradient(f, vars)` | Gradient vector |
| `Jacobian` | `Jacobian(F, vars)` | Jacobian matrix |
| `Hessian` | `Hessian(f, vars)` | Hessian matrix |

### Derivative at a Point

| Function | Signature | Description |
|----------|-----------|-------------|
| `DerivAt` | `DerivAt(f, a, n?)` | f^(n)(a) |
| `DiffQuotient` | `DiffQuotient(f, a, h)` | (f(a+h) - f(a))/h |
| `SymDiff` | `SymDiff(f, a, h)` | Symmetric difference quotient |

### Numeric Differentiation Methods

```
DerivAt(f, a, {
  method: "central",      # "forward", "backward", "central", "richardson"
  h: 1/1000,              # Step size (or auto)
  order: 4,               # Richardson extrapolation order
  precision: 1/1000000    # Target precision
})
```

### Symbolic Differentiation

When `f` has symbolic representation:

```
f := x -> x^3 + 2*x
Set(f, "symbolic", "x^3 + 2*x")

Derivative(f)             # Returns g where g(x) = 3*x^2 + 2
                          # With Set(g, "symbolic", "3*x^2 + 2")
```

**Differentiation Rules** (implemented symbolically):
- Power rule: d/dx[x^n] = n·x^(n-1)
- Sum rule: d/dx[f + g] = f' + g'
- Product rule: d/dx[f·g] = f'·g + f·g'
- Quotient rule: d/dx[f/g] = (f'·g - f·g')/g²
- Chain rule: d/dx[f(g(x))] = f'(g(x))·g'(x)
- Transcendental: d/dx[sin], d/dx[exp], d/dx[ln], etc.

---

## Category 3: Integrals

### Definite Integrals

| Function | Signature | Description |
|----------|-----------|-------------|
| `Integral` | `Integral(f, x, a, b, opts?)` | ∫_a^b f(x) dx |
| `Int` | `Int(f, a, b)` | Shorthand for definite integral |

### Indefinite Integrals (Symbolic)

| Function | Signature | Description |
|----------|-----------|-------------|
| `Antiderivative` | `Antiderivative(f, x)` | Find F where F' = f |
| `Integrate` | `Integrate(f, x)` | Symbolic integration |

### Numeric Integration Methods

```
Integral(f, x, 0, 1, {
  method: "adaptive",     # See below
  precision: 1/1000000,   # Target precision
  maxEvals: 10000,        # Max function evaluations
  subdivisions: 100       # For non-adaptive methods
})
```

**Methods:**
- `"trapezoid"`: Trapezoidal rule
- `"simpson"`: Simpson's rule
- `"romberg"`: Romberg integration
- `"adaptive"`: Adaptive quadrature (default)
- `"gausslegendre"`: Gauss-Legendre quadrature
- `"montecarlo"`: Monte Carlo (for high dimensions)

### Improper Integrals

| Function | Signature | Description |
|----------|-----------|-------------|
| `ImproperInt` | `ImproperInt(f, x, a, b)` | Handle infinite limits/singularities |
| `Converges` | `Converges(integral)` | Check if integral converges |

### Multiple Integrals

| Function | Signature | Description |
|----------|-----------|-------------|
| `DoubleInt` | `DoubleInt(f, x, a, b, y, c, d)` | Double integral |
| `TripleInt` | `TripleInt(f, ...)` | Triple integral |
| `LineInt` | `LineInt(F, curve)` | Line integral |
| `SurfaceInt` | `SurfaceInt(F, surface)` | Surface integral |

---

## Category 4: Series

### Summation

| Function | Signature | Description |
|----------|-----------|-------------|
| `Sum` | `Sum(expr, n, a, b)` | Σ_{n=a}^{b} expr |
| `InfiniteSum` | `InfiniteSum(expr, n, a)` | Σ_{n=a}^{∞} expr |
| `Product` | `Product(expr, n, a, b)` | Π_{n=a}^{b} expr |
| `InfiniteProduct` | `InfiniteProduct(expr, n, a)` | Π_{n=a}^{∞} expr |

### Power Series

| Function | Signature | Description |
|----------|-----------|-------------|
| `TaylorSeries` | `TaylorSeries(f, x, a, n)` | Taylor series at a, n terms |
| `MaclaurinSeries` | `MaclaurinSeries(f, x, n)` | Taylor at 0 |
| `TaylorCoeffs` | `TaylorCoeffs(f, x, a, n)` | Taylor coefficients |
| `RadiusConv` | `RadiusConv(series)` | Radius of convergence |

### Series Convergence

| Function | Signature | Description |
|----------|-----------|-------------|
| `Converges` | `Converges(series)` | Check if series converges |
| `RatioTest` | `RatioTest(series)` | Apply ratio test |
| `RootTest` | `RootTest(series)` | Apply root test |
| `IntegralTest` | `IntegralTest(series)` | Apply integral test |
| `ComparisonTest` | `ComparisonTest(s1, s2)` | Compare two series |

### Common Series

| Function | Signature | Description |
|----------|-----------|-------------|
| `GeometricSeries` | `GeometricSeries(a, r, n?)` | a + ar + ar² + ... |
| `HarmonicSeries` | `HarmonicSeries(n?)` | 1 + 1/2 + 1/3 + ... |
| `PSeriesSum` | `PSeriesSum(p, n?)` | Σ 1/n^p |
| `AlternatingSeries` | `AlternatingSeries(a, n?)` | Alternating series |

---

## Category 5: Differential Equations

### First-Order ODEs

| Function | Signature | Description |
|----------|-----------|-------------|
| `ODESolve` | `ODESolve(eq, y, x, init)` | Solve y' = f(x, y) |
| `SlopeField` | `SlopeField(f, xrange, yrange)` | Generate slope field |
| `EulerMethod` | `EulerMethod(f, x0, y0, h, n)` | Euler's method |
| `RK4` | `RK4(f, x0, y0, h, n)` | Runge-Kutta 4th order |

### Separable & Linear

| Function | Signature | Description |
|----------|-----------|-------------|
| `SeparableODE` | `SeparableODE(g, h, x, y)` | Solve y' = g(x)h(y) |
| `LinearODE1` | `LinearODE1(p, q, x, y)` | Solve y' + p(x)y = q(x) |

### Second-Order ODEs

| Function | Signature | Description |
|----------|-----------|-------------|
| `LinearODE2` | `LinearODE2(a, b, c, f, x, y)` | ay'' + by' + cy = f(x) |
| `CharEq` | `CharEq(a, b, c)` | Characteristic equation |

---

## Category 6: Optimization

### Finding Extrema

| Function | Signature | Description |
|----------|-----------|-------------|
| `FindMin` | `FindMin(f, x, a, b)` | Find minimum in [a, b] |
| `FindMax` | `FindMax(f, x, a, b)` | Find maximum in [a, b] |
| `CriticalPoints` | `CriticalPoints(f, x, a, b)` | Find where f'(x) = 0 |
| `Inflection` | `Inflection(f, x, a, b)` | Find inflection points |
| `Extrema` | `Extrema(f, x, a, b)` | All local extrema |

### Root Finding (related)

| Function | Signature | Description |
|----------|-----------|-------------|
| `FindRoot` | `FindRoot(f, x, guess)` | Newton's method |
| `Bisection` | `Bisection(f, a, b, tol)` | Bisection method |
| `Secant` | `Secant(f, x0, x1, tol)` | Secant method |

### Kantorovich's Theorem

For Newton's method convergence, when f has known derivative bounds:

```
# If f has:
# - |f'(x)| ≥ m > 0 on interval
# - |f''(x)| ≤ M on interval  
# - |f(x₀)/f'(x₀)| ≤ h
# Then Newton's method converges if h·M/m ≤ 1/2

KantorovichCheck(f, x0, interval)
# Returns: {converges: true/false, h, m, M, bound}
```

**This is a key theorem for rigorous root finding with oracles.**

---

## Category 7: Real Interface Abstraction

### Interface Definition

```javascript
// All real implementations provide:
const RealInterface = {
  // Narrow to interval of width ≤ epsilon
  narrow: (epsilon) => RationalInterval,
  
  // Arithmetic
  add: (other) => Real,
  sub: (other) => Real,
  mul: (other) => Real,
  div: (other) => Real,
  neg: () => Real,
  abs: () => Real,
  
  // Comparison with rational
  lt: (q) => boolean | undefined,
  gt: (q) => boolean | undefined,
  eq: (q) => boolean | undefined,
  
  // Optional symbolic
  hasDerivative: boolean,
  derivative: () => Real | undefined,
};
```

### Adapter Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `AsReal` | `AsReal(value, type?)` | Convert to Real interface |
| `FromOracle` | `FromOracle(oracle)` | Oracle → Real |
| `FromInterval` | `FromInterval(interval)` | Interval → Real |
| `FromCauchy` | `FromCauchy(seq)` | Cauchy sequence → Real |
| `ToOracle` | `ToOracle(real)` | Real → Oracle |
| `RealType` | `RealType(real)` | Get underlying type |

### Real Arithmetic Layer

```
# These work regardless of real representation:
RealAdd(a, b)
RealMul(a, b)
RealNarrow(a, epsilon)
RealCompare(a, b, epsilon)  # Returns -1, 0, 1, or undefined
```

---

## Algorithms Leveraging Derivatives

When functions have known derivatives, special algorithms become available:

### 1. Newton's Method with Kantorovich

```
NewtonKantorovich(f, x0, {
  derivatives: {f, f', f''},  # If provided
  interval: [a, b],           # Search interval
  tolerance: 1e-10
})
# Rigorous convergence guarantee when conditions met
```

### 2. Automatic Differentiation

If derivatives are computed symbolically or via AD:

```
AutoDiff(f, x)                # Returns {value, derivative}
TaylorAD(f, x, n)             # Taylor coefficients via AD
```

### 3. Verified Integration

Using derivative bounds for error control:

```
VerifiedIntegral(f, a, b, {
  derivativeBound: M,         # |f'| ≤ M on [a, b]
  tolerance: epsilon
})
# Returns interval guaranteed to contain true value
```

### 4. Interval Newton Method

```
IntervalNewton(f, interval, {
  f': derivative,             # Interval extension of f'
  maxIter: 100
})
# Returns enclosure of all roots in interval
```

---

## Implementation Priority

### Phase 1: Basic Derivatives
- [ ] Numeric differentiation (central difference)
- [ ] DerivAt function
- [ ] Basic symbolic rules (power, sum)

### Phase 2: Basic Integrals
- [ ] Simpson's rule
- [ ] Adaptive quadrature
- [ ] Definite integral interface

### Phase 3: Limits
- [ ] Sequence-based limits
- [ ] One-sided limits
- [ ] Richardson extrapolation

### Phase 4: Real Interface
- [ ] Define interface
- [ ] Oracle adapter
- [ ] Interval adapter

### Phase 5: Series
- [ ] Finite sums
- [ ] Taylor series
- [ ] Convergence tests

### Phase 6: Root Finding
- [ ] Bisection
- [ ] Newton's method
- [ ] Kantorovich conditions

### Phase 7: ODEs
- [ ] Euler method
- [ ] RK4
- [ ] Slope field generation

### Phase 8: Symbolic
- [ ] Symbolic differentiation
- [ ] Basic symbolic integration
- [ ] Automatic differentiation

---

## Dependencies

- `@ratmath/core`: Rational, RationalInterval
- `@ratmath/oracles`: Oracle real number interface
- `@ratmath/arith-funs`: Polynomials (for Taylor)
- `@ratmath/algebra`: Symbolic expression handling

---

## Open Questions

1. **Primary real type**: Which real implementation is default?
   - Proposed: Oracles, with interval fallback

2. **Symbolic depth**: How sophisticated should symbolic calculus be?
   - Proposed: Basic rules + hooks for user extensions

3. **Verified vs fast**: Always rigorous or offer fast approximations?
   - Proposed: Rigorous by default, `fast: true` option

4. **Complex calculus**: Support complex analysis?
   - Proposed: Not initially; separate extension later

5. **Automatic differentiation**: Forward or reverse mode?
   - Proposed: Forward mode for simplicity; reverse for gradients later
