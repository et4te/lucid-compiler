# Lucid Dataflow Programming Language Compiler

A comprehensive implementation of a compiler for the Lucid dataflow programming language in Rust, featuring:

- **Demand-driven evaluation** with lazy stream processing
- **Temporal operators** (`fby`, `next`, `prev`, `first`)
- **Higher-order functions** with lambda expressions
- **Advanced optimizations**:
  - Common Subexpression Elimination (CSE)
  - Demand Analysis
  - Loop Fusion
  - Buffer Minimization

## Architecture

```
Source Code
    ↓
[Lexer] → Tokens
    ↓
[Parser] → AST
    ↓
[Graph Builder] → Dataflow Graph
    ↓
[Optimizer] → Optimized Graph
    ├── CSE Pass
    ├── Demand Analysis
    ├── Loop Fusion
    └── Buffer Minimization
    ↓
[Evaluator] → Results (with memoization)
```

## Language Features

### Temporal Operators

**fby (followed-by)**: The fundamental temporal operator
```
fib = 0 fby 1 fby fib + next fib
```

**next**: Look ahead one time step
```
x = next y
```

**prev**: Look back one time step
```
x = prev y
```

**first**: Extract the initial value
```
x = first stream
```

### Higher-Order Functions

Lambda expressions with function application:
```
square = fn(x) -> x * x;
result = square(5)
```

### Control Flow

If-then-else expressions:
```
abs = if x < 0 then -x else x
```

### Local Bindings

Let expressions:
```
let x = 5, y = 10 in x + y
```

Where clauses (for recursive definitions):
```
fib where {
    fib = 0 fby 1 fby fib + next fib
}
```

## Optimization Techniques

### 1. Common Subexpression Elimination (CSE)

Identifies and eliminates duplicate computations by sharing results:

```rust
// Before CSE:
// (a + b) * 2 + (a + b) * 3

// After CSE:
// temp = a + b
// temp * 2 + temp * 3
```

**Implementation**: Uses expression signatures to identify duplicates and creates a mapping from redundant nodes to canonical nodes.

### 2. Demand Analysis

Determines which values are actually needed for computation:

```rust
// Only computes nodes reachable from entry point
// Eliminates dead code
// Calculates how many future values each stream needs
```

**Benefits**:
- Dead code elimination
- Minimal computation
- Optimal lazy evaluation

### 3. Loop Fusion

Combines multiple passes over streams into single passes:

```rust
// Before fusion:
// for t in 0..n: compute stream_a[t]
// for t in 0..n: compute stream_b[t]

// After fusion:
// for t in 0..n: 
//     compute stream_a[t]
//     compute stream_b[t]
```

**Benefits**:
- Reduced iteration overhead
- Better cache locality
- Improved performance

### 4. Buffer Minimization

Calculates minimum buffer sizes needed for temporal operations:

```rust
// Analyzes delay chains to determine minimum storage
// Prevents over-allocation
// Optimizes memory usage
```

## Demand-Driven Evaluation

The evaluator uses **memoization** and **lazy evaluation**:

1. Values are only computed when demanded
2. Once computed, results are cached
3. Temporal operators access cached historical values
4. Efficient for sparse access patterns

### Memoization Strategy

```rust
cache: HashMap<(NodeId, TimeStep), Value>

fn eval(node, time):
    if cached(node, time):
        return cache[(node, time)]
    
    result = compute(node, time)
    cache[(node, time)] = result
    return result
```

## Usage Examples

### Basic Usage

```rust
use lucid_compiler::LucidCompiler;

let mut compiler = LucidCompiler::new();

// Compile a program
compiler.compile("(5 + 3) * 2").unwrap();

// Evaluate for 1 time step
let result = compiler.eval(1).unwrap();
println!("Result: {:?}", result);

// Print optimization statistics
compiler.print_optimization_stats();
```

### Fibonacci Sequence

```rust
let mut compiler = LucidCompiler::new();

// Simple two-element stream
compiler.compile("0 fby 1").unwrap();

// Evaluate for 10 time steps
let values = compiler.eval(10).unwrap();
// Output: [0, 1, 1, 1, 1, ...]
```

### With Optimizations

```rust
let mut compiler = LucidCompiler::new();

// Expression with common subexpressions
compiler.compile("(5 + 3) + (5 + 3)").unwrap();

// The optimizer will:
// 1. Identify (5 + 3) appears twice
// 2. Create single computation
// 3. Reuse result
```

## Building and Testing

```bash
# Build the project
cargo build --release

# Run tests
cargo test

# Run examples
cargo run

# Run with verbose output
cargo run -- --verbose
```

## Implementation Details

### Module Structure

- `lexer.rs`: Tokenization using `logos`
- `parser.rs`: Parsing using `chumsky`
- `ast.rs`: Abstract syntax tree definitions
- `dataflow.rs`: Dataflow graph IR
- `graph_builder.rs`: AST → Graph conversion
- `demand_analysis.rs`: Demand analysis pass
- `cse.rs`: Common subexpression elimination
- `loop_fusion.rs`: Loop fusion and buffer minimization
- `evaluator.rs`: Demand-driven evaluator
- `optimizer.rs`: Orchestrates all optimization passes

### Key Data Structures

**DataflowGraph**: Intermediate representation
```rust
pub struct DataflowGraph {
    nodes: HashMap<NodeId, Node>,
    edges: HashMap<NodeId, Vec<Edge>>,
    entry_point: NodeId,
}
```

**Node**: Operations in the dataflow graph
```rust
pub enum Node {
    Constant(Value),
    BinOp { op, left, right },
    Delay { input, amount, init },
    Select { cond, then_val, else_val },
    Lambda { params, body },
    Apply { func, args },
    // ...
}
```

## Performance Characteristics

### Time Complexity

- **CSE**: O(n) where n is number of nodes
- **Demand Analysis**: O(n + e) where e is edges
- **Loop Fusion**: O(n²) worst case (with merging)
- **Buffer Minimization**: O(n × d) where d is max delay
- **Evaluation**: O(t × k) where t is time steps, k is demanded nodes

### Space Complexity

- **Memoization cache**: O(t × k) where t is time steps
- **Optimized buffers**: O(k × b) where b is avg buffer size
- **Graph storage**: O(n + e)

## Future Enhancements

- [ ] Incremental evaluation
- [ ] Parallel evaluation of independent streams
- [ ] JIT compilation to native code
- [ ] Stream windowing operators
- [ ] External data source integration
- [ ] Interactive REPL
- [ ] Visual dataflow graph debugging
- [ ] Advanced type system
- [ ] Automatic parallelization

## References

- Wadge, W. W., & Ashcroft, E. A. (1985). *Lucid, the Dataflow Programming Language*
- Plaice, J., & Wadge, W. W. (1993). *A New Approach to Version Control*
- Ashcroft, E. A., Faustini, A. A., Jagannathan, R., & Wadge, W. W. (1995). *Multidimensional Programming*

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

---

**Note**: This is a educational/research implementation demonstrating advanced compiler optimization techniques for dataflow languages.
