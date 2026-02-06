# Lucid Compiler Usage Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Language Tutorial](#language-tutorial)
4. [Advanced Features](#advanced-features)
5. [Optimization Guide](#optimization-guide)
6. [API Reference](#api-reference)
7. [Examples](#examples)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lucid-compiler
cd lucid-compiler

# Build in release mode
cargo build --release

# Run tests
cargo test

# Run examples
cargo run --example fibonacci
```

## Quick Start

### Hello World

```rust
use lucid_compiler::LucidCompiler;

fn main() {
    let mut compiler = LucidCompiler::new();
    
    // Compile a simple expression
    compiler.compile("1 + 2").unwrap();
    
    // Evaluate for 1 time step
    let result = compiler.eval(1).unwrap();
    
    println!("Result: {:?}", result); // [3]
}
```

### Your First Stream

```rust
let mut compiler = LucidCompiler::new();

// Create a stream with fby (followed-by)
compiler.compile("0 fby 1").unwrap();

// Evaluate for 10 time steps
let stream = compiler.eval(10).unwrap();

println!("Stream: {:?}", stream); // [0, 1, 1, 1, ...]
```

## Language Tutorial

### Basic Expressions

```lucid
// Arithmetic
5 + 3 * 2        // 11

// Comparison
10 > 5           // true

// Boolean logic
true && false    // false

// Nested expressions
(5 + 3) * (10 - 2)  // 64
```

### Variables and Bindings

```lucid
// Let expressions
let x = 5, y = 10 in x + y

// Where clauses
x + y where {
    x = 5,
    y = 10
}
```

### Conditional Expressions

```lucid
// If-then-else
if x > 0 then x else -x

// Nested conditionals
if x > 0 then
    if x > 10 then 100 else x
else
    -x
```

### Temporal Operators

#### The fby Operator

`fby` is the fundamental temporal operator in Lucid.

```lucid
// Basic usage: init fby next
0 fby 1         // Stream: [0, 1, 1, 1, ...]

// Chained fby
0 fby 1 fby 2   // Stream: [0, 1, 2, 2, 2, ...]

// With expressions
init fby (next + 1)
```

**How fby works:**
- At time 0: returns the init value
- At time t > 0: returns the next value at time t-1

#### The next Operator

```lucid
// Look ahead one time step
next x

// Example: increment stream
x where {
    x = 0 fby next x + 1
}
// Stream: [0, 1, 2, 3, 4, ...]
```

#### The prev Operator

```lucid
// Look back one time step
prev x

// Example: running sum
sum where {
    sum = x fby sum + x
}
```

#### The first Operator

```lucid
// Extract initial value
first stream
```

### Higher-Order Functions

#### Lambda Expressions

```lucid
// Simple lambda
fn(x) -> x + 1

// Multiple parameters
fn(x, y) -> x + y

// Nested lambdas
fn(x) -> fn(y) -> x + y
```

#### Function Application

```lucid
// Direct application
(fn(x) -> x * 2)(5)  // 10

// Stored function
square = fn(x) -> x * x;
square(5)  // 25

// Higher-order usage
map = fn(f, x) -> f(x);
map(fn(x) -> x + 1, 5)  // 6
```

### Stream Processing

#### Fibonacci Sequence

Classic example using mutual recursion:

```lucid
fib where {
    fib = 0 fby 1 fby fib + next fib
}
```

Breakdown:
1. First value: 0
2. Second value: 1
3. Each subsequent: sum of previous two

#### Counter

```lucid
count where {
    count = 0 fby count + 1
}
// Stream: [0, 1, 2, 3, 4, ...]
```

#### Running Average

```lucid
avg where {
    sum = x fby sum + x,
    count = 1 fby count + 1,
    avg = sum / count
}
```

## Advanced Features

### Mutual Recursion

```lucid
even_odd where {
    even = true fby odd,
    odd = false fby even
}
```

### Stream Transformations

```lucid
// Filter (using conditional)
filtered where {
    filtered = if x > 0 then x else prev filtered
}

// Map
mapped where {
    mapped = fn(f) -> f(x)
}

// Fold/Reduce
sum where {
    sum = x fby sum + x
}
```

### Working with Time

```lucid
// Delayed stream
delayed where {
    delayed = 0 fby 0 fby x
}
// Delays x by 2 time steps

// Synchronized streams
synced where {
    a = x fby a + 1,
    b = y fby b * 2,
    synced = a + b
}
```

## Optimization Guide

### Understanding Optimizations

The compiler applies several optimization passes automatically:

1. **Common Subexpression Elimination (CSE)**
2. **Demand Analysis**
3. **Loop Fusion**
4. **Buffer Minimization**

### Viewing Optimization Results

```rust
let mut compiler = LucidCompiler::new();
compiler.compile("(5 + 3) + (5 + 3)").unwrap();

// Print optimization statistics
compiler.print_optimization_stats();
```

Output:
```
=== Optimization Statistics ===
Demanded nodes: 3
Total nodes: 3
Fusion groups: 1
  Group 0: 3 nodes
Buffer requirements:
  Node 0: 1 values
  Node 1: 1 values
  Node 2: 1 values
CSE eliminations: 1
```

### Writing Optimization-Friendly Code

#### Good Practices

```lucid
// Share common subexpressions
let common = expensive_calc in
    common + common

// Instead of:
expensive_calc + expensive_calc
```

```lucid
// Use where for mutual recursion
fib where {
    fib = 0 fby 1 fby fib + next fib
}

// Instead of multiple definitions
```

#### Avoiding Common Pitfalls

```lucid
// ❌ Avoid: Redundant delays
x fby x fby x fby ...

// ✅ Better: Single delay chain
x fby y

// ❌ Avoid: Deep nesting
if a then (if b then (if c then ...))

// ✅ Better: Flatten when possible
if a && b && c then ...
```

### Performance Tips

1. **Minimize Demanded Depth**: Shorter delay chains = less buffering
2. **Share Computations**: Use let/where to avoid recomputation
3. **Avoid Unnecessary Delays**: Each delay requires buffering
4. **Use Primitive Operations**: Built-in ops are optimized

## API Reference

### LucidCompiler

Main compiler interface.

```rust
pub struct LucidCompiler { /* ... */ }

impl LucidCompiler {
    // Create new compiler
    pub fn new() -> Self;
    
    // Compile source code
    pub fn compile(&mut self, source: &str) -> Result<(), CompileError>;
    
    // Evaluate for n time steps
    pub fn eval(&self, steps: usize) -> Result<Vec<Value>, EvalError>;
    
    // Get optimized dataflow graph
    pub fn get_graph(&self) -> Option<&DataflowGraph>;
    
    // Print optimization statistics
    pub fn print_optimization_stats(&self);
}
```

### Value Types

```rust
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Stream(Vec<Value>),
}
```

### Error Types

```rust
pub enum CompileError {
    LexError(String),
    ParseError(String),
    GraphBuildError(String),
}

pub enum EvalError {
    NodeNotFound(NodeId),
    UndefinedInput(String),
    TypeError(String),
    DivisionByZero,
    NotAFunction,
    ArityMismatch { expected: usize, got: usize },
}
```

## Examples

### Example 1: Natural Numbers

```rust
use lucid_compiler::LucidCompiler;

fn natural_numbers() {
    let mut compiler = LucidCompiler::new();
    
    let source = r#"
        naturals where {
            naturals = 1 fby naturals + 1
        }
    "#;
    
    compiler.compile(source).unwrap();
    let numbers = compiler.eval(10).unwrap();
    
    println!("Natural numbers: {:?}", numbers);
    // [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
```

### Example 2: Factorial Stream

```rust
fn factorials() {
    let mut compiler = LucidCompiler::new();
    
    let source = r#"
        fact where {
            n = 1 fby n + 1,
            fact = 1 fby fact * n
        }
    "#;
    
    compiler.compile(source).unwrap();
    let facts = compiler.eval(7).unwrap();
    
    println!("Factorials: {:?}", facts);
    // [1, 1, 2, 6, 24, 120, 720]
}
```

### Example 3: Prime Number Sieve

```rust
fn primes() {
    let mut compiler = LucidCompiler::new();
    
    let source = r#"
        candidates where {
            candidates = 2 fby candidates + 1
        }
    "#;
    
    compiler.compile(source).unwrap();
    let nums = compiler.eval(20).unwrap();
    
    println!("Candidate numbers: {:?}", nums);
}
```

### Example 4: Moving Average

```rust
fn moving_average() {
    let mut compiler = LucidCompiler::new();
    
    let source = r#"
        avg where {
            window = x + prev x + prev prev x,
            avg = window / 3
        }
    "#;
    
    // Note: This is simplified; full implementation
    // would require input handling
}
```

### Example 5: State Machine

```rust
fn state_machine() {
    let mut compiler = LucidCompiler::new();
    
    let source = r#"
        state where {
            state = 0 fby next_state,
            next_state = if state == 0 then 1
                        else if state == 1 then 2
                        else 0
        }
    "#;
    
    compiler.compile(source).unwrap();
    let states = compiler.eval(10).unwrap();
    
    println!("State sequence: {:?}", states);
}
```

## Troubleshooting

### Common Errors

#### Parse Error

```
Error: Parse error: Unexpected token 'fby' at position 10
```

**Solution**: Check syntax, ensure operators are properly spaced.

#### Type Error

```
Error: Type error: Expected boolean in condition
```

**Solution**: Ensure if conditions evaluate to boolean.

#### Undefined Input

```
Error: Undefined input: x
```

**Solution**: All variables must be defined or bound.

### Performance Issues

If evaluation is slow:

1. Check demanded depth (using `print_optimization_stats`)
2. Look for deep recursion
3. Consider reducing time steps
4. Profile with smaller inputs first

## Best Practices

1. **Start Simple**: Build complexity incrementally
2. **Test Often**: Verify each component works
3. **Use Where**: For mutually recursive definitions
4. **Leverage Optimizations**: Write code that optimizes well
5. **Check Statistics**: Monitor optimization impact

## Contributing

See CONTRIBUTING.md for guidelines on:
- Code style
- Testing requirements
- Documentation standards
- Pull request process

## License

MIT License - see LICENSE file for details.
