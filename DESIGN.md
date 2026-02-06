# Lucid Compiler Design Documentation

## Overview

This document describes the design and implementation of a production-quality Lucid dataflow programming language compiler with advanced optimization capabilities.

## Architecture

### 1. Lexical Analysis (lexer.rs)

**Technology**: Uses the `logos` crate for efficient lexical analysis.

**Design Decisions**:
- Regex-based token matching for performance
- Skip whitespace and comments automatically
- Separate tokens for all temporal operators
- Rich literal support (integers, floats, booleans)

**Key Features**:
```rust
#[derive(Logos, Debug, Clone, PartialEq)]
pub enum Token {
    #[token("fby")] Fby,
    #[token("next")] Next,
    #[regex(r"[0-9]+", |lex| lex.slice().parse())] Integer(i64),
    // ...
}
```

### 2. Parsing (parser.rs)

**Technology**: Uses the `chumsky` parser combinator library.

**Design Decisions**:
- Recursive descent parsing with combinator style
- Operator precedence handled via nested parsers
- Right-associative `fby` operator (critical for temporal logic)
- Support for lambda expressions and higher-order functions

**Precedence Levels** (lowest to highest):
1. `fby` - right associative
2. Logical operators (`&&`, `||`)
3. Comparison operators (`==`, `!=`, `<`, `>`, etc.)
4. Additive operators (`+`, `-`)
5. Multiplicative operators (`*`, `/`)
6. Unary operators (`-`, `!`)
7. Function application
8. Atoms (constants, variables, parenthesized expressions)

### 3. Abstract Syntax Tree (ast.rs)

**Design Philosophy**: Simple, composable expression types.

**Key Design Decisions**:

1. **Unified Expression Type**: All constructs are expressions
2. **Temporal Operators as First-Class**: `Fby`, `Next`, `Prev`, `First`
3. **Higher-Order Functions**: Lambda and App nodes
4. **Local Scoping**: `Let` and `Where` for bindings

**Expression Variants**:
```rust
pub enum Expr {
    // Values
    Int(i64), Float(f64), Bool(bool), Var(Ident),
    
    // Temporal
    Fby { init: Box<Expr>, next: Box<Expr> },
    Next(Box<Expr>), Prev(Box<Expr>), First(Box<Expr>),
    
    // Operations
    BinOp { op, left, right },
    UnOp { op, expr },
    
    // Control
    If { cond, then_branch, else_branch },
    
    // Functions
    Lambda { params, body },
    App { func, args },
    
    // Scoping
    Let { bindings, body },
    Where { expr, bindings },
}
```

### 4. Dataflow Graph IR (dataflow.rs)

**Purpose**: Intermediate representation optimized for analysis and transformation.

**Key Design Decisions**:

1. **Graph-Based**: Nodes and edges explicitly represent data dependencies
2. **Temporal Edges**: Edges carry delay information
3. **Immutable Values**: Values are immutable for safe memoization
4. **Explicit Dependencies**: Easy to traverse and analyze

**Node Types**:
```rust
pub enum Node {
    Constant(Value),
    Input(String),
    BinOp { op, left, right },
    UnOp { op, operand },
    Delay { input, amount, init },
    Select { cond, then_val, else_val },
    Lambda { params, body },
    Apply { func, args },
    Memo { expr },
}
```

**Why This Design?**
- Enables efficient graph algorithms
- Separates concerns (syntax vs semantics)
- Facilitates optimization passes
- Makes dataflow explicit

### 5. Graph Builder (graph_builder.rs)

**Purpose**: Convert AST to dataflow graph.

**Design Decisions**:

1. **Environment Tracking**: Maintains variable bindings during conversion
2. **Recursive Handling**: Special care for `where` clauses with mutual recursion
3. **Closure Conversion**: Captures free variables in lambdas
4. **Temporal Lowering**: Converts high-level temporal ops to graph primitives

**Key Algorithm**:
```rust
fn build_expr(&mut self, expr: &Expr) -> NodeId {
    match expr {
        Expr::Fby { init, next } => {
            let init_node = self.build_expr(init);
            let next_node = self.build_expr(next);
            
            // Extract constant init value
            let init_val = extract_constant(init_node);
            
            // Create delay node
            self.graph.add_node(Node::Delay {
                input: next_node,
                amount: 1,
                init: init_val,
            })
        }
        // ...
    }
}
```

### 6. Demand Analysis (demand_analysis.rs)

**Purpose**: Determine which nodes are actually needed.

**Algorithm**: Backward reachability from entry point.

**Key Insights**:

1. **Depth Tracking**: Track how many future values are needed
2. **Temporal Awareness**: Delays increase demand depth
3. **Conservative for Branches**: Both branches of `if` are demanded
4. **Dead Code Identification**: Unreachable nodes can be eliminated

**Optimization Impact**:
- Eliminates unused computations
- Determines buffer sizes
- Enables lazy evaluation

**Implementation**:
```rust
pub struct DemandAnalysis {
    pub demanded: HashSet<NodeId>,
    pub demand_depth: HashMap<NodeId, usize>,
}

impl DemandAnalysis {
    fn mark_demanded(&mut self, graph: &DataflowGraph, node_id: NodeId, depth: usize) {
        match graph.nodes.get(&node_id) {
            Some(Node::Delay { input, amount, .. }) => {
                // Need depth + amount values from input
                self.mark_demanded(graph, *input, depth + amount);
            }
            // ...
        }
    }
}
```

### 7. Common Subexpression Elimination (cse.rs)

**Purpose**: Eliminate redundant computations.

**Algorithm**: Hash-consing with expression signatures.

**Key Design Decisions**:

1. **Signature-Based**: Uses structural equality
2. **Topological Processing**: Process dependencies first
3. **Mapping Table**: Track old → new node mappings
4. **Conservative**: Only eliminates pure operations

**Expression Signatures**:
```rust
#[derive(Hash, Eq, PartialEq)]
enum ExprSignature {
    Constant(String),
    BinOp { op, left: NodeId, right: NodeId },
    UnOp { op, operand: NodeId },
    Delay { input: NodeId, amount: usize },
    // ...
}
```

**Process**:
1. Compute signature for each node
2. Check cache for existing computation
3. If found, create mapping
4. If not, cache this node
5. Apply mappings to update all references

**Optimization Impact**:
- Reduces redundant arithmetic
- Shares common delay operations
- Can combine with demand analysis for further reduction

### 8. Loop Fusion (loop_fusion.rs)

**Purpose**: Combine multiple stream iterations into single passes.

**Algorithm**: Fusion group identification based on temporal boundaries.

**Key Concepts**:

1. **Fusion Groups**: Sets of nodes computed together
2. **Temporal Boundaries**: Delay nodes define iteration boundaries
3. **Group Merging**: Overlapping groups are combined
4. **Scheduling**: Determines execution order

**Algorithm**:
```rust
1. Identify all temporal nodes (delays)
2. For each temporal node:
   - Build fusion group of reachable non-temporal nodes
3. Merge overlapping groups
4. Generate schedule
```

**Benefits**:
- Reduces loop overhead
- Improves cache locality
- Enables vectorization opportunities

### 9. Buffer Minimization (loop_fusion.rs)

**Purpose**: Calculate minimum buffer sizes for streams.

**Algorithm**: Analyze delay chains within fusion groups.

**Key Insights**:

1. **Delay Chain Analysis**: Track cumulative delays
2. **Group-Based**: Calculate per fusion group
3. **Conservative**: Ensures correctness

**Buffer Size Calculation**:
```
buffer_size(node) = max_delay_to_node + 1
```

**Process**:
1. For each fusion group:
   - Calculate maximum delay to each node
   - Buffer size = max_delay + 1 (for current value)
2. Aggregate results

**Optimization Impact**:
- Minimizes memory usage
- Prevents over-allocation
- Enables efficient circular buffers

### 10. Demand-Driven Evaluation (evaluator.rs)

**Purpose**: Lazy evaluation with memoization.

**Design Philosophy**: Compute only what's needed, when needed, once.

**Key Components**:

1. **Memoization Cache**: `(NodeId, TimeStep) → Value`
2. **Recursive Evaluation**: Demand propagates through graph
3. **Temporal Semantics**: Delays access earlier time steps
4. **Environment Management**: For lambda evaluation

**Evaluation Algorithm**:
```rust
fn eval(node_id, time):
    // Check cache
    if cached(node_id, time):
        return cache[(node_id, time)]
    
    // Evaluate based on node type
    result = match node:
        Constant(v) => v
        BinOp { op, left, right } =>
            apply_op(op, eval(left, time), eval(right, time))
        Delay { input, amount, init } =>
            if time < amount:
                init
            else:
                eval(input, time - amount)
        // ...
    
    // Memoize and return
    cache[(node_id, time)] = result
    result
```

**Key Features**:

1. **Lazy**: Only compute demanded values
2. **Memoized**: Cache all computed values
3. **Temporal**: Handle delays correctly
4. **Higher-Order**: Support function application

**Performance Characteristics**:
- Time: O(demanded_nodes × time_steps)
- Space: O(demanded_nodes × time_steps) for cache
- With optimizations: Often much better in practice

### 11. Optimizer Orchestration (optimizer.rs)

**Purpose**: Coordinate all optimization passes.

**Pass Ordering** (carefully chosen):

1. **CSE** - Eliminate duplicates first
2. **Demand Analysis** - Identify needed nodes
3. **Dead Code Elimination** - Remove unneeded nodes
4. **Loop Fusion** - Group related computations
5. **Buffer Minimization** - Optimize memory

**Why This Order?**

1. CSE first reduces graph size for other passes
2. Demand analysis needs accurate graph structure
3. Dead code elimination simplifies remaining passes
4. Loop fusion benefits from smaller graph
5. Buffer minimization is final refinement

**Statistics Tracking**:
```rust
pub struct OptimizationResult {
    cse_eliminations: usize,
    dead_code_eliminated: usize,
    fusion_groups: usize,
    total_buffer_size: usize,
    avg_buffer_size: f64,
}
```

## Design Patterns

### 1. Visitor Pattern

Used throughout for graph traversal:
```rust
fn visit_node(&mut self, node_id: NodeId) {
    match self.graph.nodes.get(&node_id) {
        Some(Node::BinOp { left, right, .. }) => {
            self.visit_node(*left);
            self.visit_node(*right);
        }
        // ...
    }
}
```

### 2. Builder Pattern

For graph construction:
```rust
pub struct GraphBuilder {
    graph: DataflowGraph,
    env: HashMap<String, NodeId>,
}
```

### 3. Strategy Pattern

Different evaluation strategies possible:
- Demand-driven (implemented)
- Data-driven (future)
- Hybrid (future)

### 4. Cache Pattern

Memoization throughout:
- Evaluation cache
- CSE expression cache
- Demand analysis results

## Performance Considerations

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Lexing | O(n) | n = source length |
| Parsing | O(n) | Linear with backtracking |
| Graph Building | O(n) | n = AST nodes |
| CSE | O(n) | n = graph nodes |
| Demand Analysis | O(n + e) | e = edges |
| Loop Fusion | O(n²) | Worst case with merging |
| Evaluation | O(t × k) | t = time, k = demanded |

### Space Complexity

| Structure | Space | Notes |
|-----------|-------|-------|
| AST | O(n) | n = nodes |
| Dataflow Graph | O(n + e) | n = nodes, e = edges |
| Memoization | O(t × k) | t = time, k = nodes |
| Optimizations | O(n) | Various caches |

### Optimization Opportunities

1. **Incremental Evaluation**: Reuse results across runs
2. **Parallel Evaluation**: Independent streams in parallel
3. **JIT Compilation**: Compile hot paths to native code
4. **Stream Fusion**: Combine operations at codegen
5. **Symbolic Execution**: Constant folding and propagation

## Testing Strategy

### Unit Tests

- Each module has dedicated tests
- Test edge cases and error conditions
- Property-based testing for graph operations

### Integration Tests

- Full compilation pipeline
- Various language features
- Optimization effectiveness

### Benchmarks

Would measure:
- Compilation time
- Evaluation performance
- Memory usage
- Optimization impact

## Future Enhancements

### Short Term

1. Better error messages with source locations
2. Type inference system
3. More temporal operators (asa, wvr, etc.)
4. Interactive REPL

### Medium Term

1. Incremental compilation
2. Module system
3. Standard library
4. Debugger with time-travel

### Long Term

1. Parallel evaluation backend
2. JIT compilation to LLVM
3. GPU acceleration for streams
4. Distributed stream processing

## Conclusion

This implementation demonstrates a complete, production-quality compiler for a dataflow language with sophisticated optimizations. The modular design allows easy extension and experimentation with new optimization techniques.
