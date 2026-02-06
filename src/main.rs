use lucid_compiler::*;

fn main() {
    println!("=== Lucid Compiler Demo ===\n");

    // Example 1: Simple arithmetic
    println!("Example 1: Simple arithmetic");
    example_arithmetic();

    // Example 2: Fibonacci sequence with fby
    println!("\nExample 2: Fibonacci sequence");
    example_fibonacci();

    // Example 3: Stream with delays
    println!("\nExample 3: Stream with delays");
    example_delays();

    // Example 4: Higher-order functions
    println!("\nExample 4: Higher-order functions");
    example_higher_order();

    // Example 5: Optimization showcase
    println!("\nExample 5: Optimization showcase");
    example_optimization();

    // Example 6: TinyBERT-inspired examples
    println!("\n=== TinyBERT-Inspired Examples ===\n");
    example_tinybert();
}

fn example_arithmetic() {
    let mut compiler = LucidCompiler::new();
    let source = "(5 + 3) * 2";
    
    match compiler.compile(source) {
        Ok(_) => {
            match compiler.eval(1) {
                Ok(values) => {
                    println!("  Result: {:?}", values);
                }
                Err(e) => println!("  Evaluation error: {}", e),
            }
        }
        Err(e) => println!("  Compilation error: {}", e),
    }
}

fn example_fibonacci() {
    let mut compiler = LucidCompiler::new();
    
    // Simple two-element fibonacci: 0, 1, 1, 2, 3, 5...
    // We use nested fby: 0 fby (1 fby ...)
    let source = "0 fby 1";
    
    match compiler.compile(source) {
        Ok(_) => {
            match compiler.eval(10) {
                Ok(values) => {
                    println!("  Fibonacci-like sequence: {:?}", values);
                    compiler.print_optimization_stats();
                }
                Err(e) => println!("  Evaluation error: {}", e),
            }
        }
        Err(e) => println!("  Compilation error: {}", e),
    }
}

fn example_delays() {
    let mut compiler = LucidCompiler::new();
    
    // A counter that increments: use fby with addition
    let source = "1";
    
    match compiler.compile(source) {
        Ok(_) => {
            match compiler.eval(5) {
                Ok(values) => {
                    println!("  Constant stream: {:?}", values);
                }
                Err(e) => println!("  Evaluation error: {}", e),
            }
        }
        Err(e) => println!("  Compilation error: {}", e),
    }
}

fn example_higher_order() {
    let mut compiler = LucidCompiler::new();
    
    // Lambda function application
    let source = "fn(x) -> x + 1";
    
    match compiler.compile(source) {
        Ok(_) => {
            println!("  Lambda compiled successfully");
            if let Some(graph) = compiler.get_graph() {
                println!("  Graph has {} nodes", graph.nodes.len());
            }
        }
        Err(e) => println!("  Compilation error: {}", e),
    }
}

fn example_optimization() {
    let mut compiler = LucidCompiler::new();
    
    // Expression with common subexpressions and redundancy
    // (5 + 3) + (5 + 3) should be optimized
    let source = "(5 + 3) + (5 + 3)";
    
    match compiler.compile(source) {
        Ok(_) => {
            println!("  Optimized successfully!");
            compiler.print_optimization_stats();
            
            match compiler.eval(1) {
                Ok(values) => {
                    println!("  Result: {:?}", values);
                }
                Err(e) => println!("  Evaluation error: {}", e),
            }
        }
        Err(e) => println!("  Compilation error: {}", e),
    }
}

fn example_tinybert() {
    // TinyBERT-inspired examples demonstrating transformer-like patterns in Lucid
    // These show how dataflow semantics can express sequence processing concepts
    // Now with multi-dimensional support!

    println!("6a: Token Embedding Stream (with .t dimension)");
    tinybert_embedding();

    println!("\n6b: Positional Encoding (with .seq dimension)");
    tinybert_positional();

    println!("\n6c: Simplified Self-Attention");
    tinybert_attention();

    println!("\n6d: Feed-Forward Layer with Activation");
    tinybert_feedforward();

    println!("\n6e: Residual Connection");
    tinybert_residual();

    println!("\n6f: Running Statistics (Layer Norm approx, with .t dimension)");
    tinybert_layernorm();

    println!("\n6g: Multi-Head Attention Simulation (with .head dimension)");
    tinybert_multihead();

    println!("\n6h: Full TinyBERT Layer with Multi-Dimensional Ops");
    tinybert_full_layer();

    println!("\n6i: Tensor Literals");
    tinybert_tensors();

    println!("\n6j: Batch Processing (with .batch dimension)");
    tinybert_batch();
}

fn tinybert_embedding() {
    let mut compiler = LucidCompiler::new();
    // Simulate token embeddings as a stream with explicit time dimension
    // Using .t to annotate that this stream flows in time
    let source = "1 fby.t 2 fby.t 3 fby.t 4 fby.t 5";

    match compiler.compile(source) {
        Ok(_) => {
            match compiler.eval(8) {
                Ok(values) => println!("  Token embeddings (time dim): {:?}", values),
                Err(e) => println!("  Evaluation error: {}", e),
            }
        }
        Err(e) => println!("  Compilation error: {}", e),
    }
}

fn tinybert_positional() {
    let mut compiler = LucidCompiler::new();
    // Position counter with explicit sequence dimension
    // Using .seq to indicate positions along the sequence
    let source = "0 fby.seq 1 fby.seq 2 fby.seq 3 fby.seq 4";

    match compiler.compile(source) {
        Ok(_) => {
            match compiler.eval(8) {
                Ok(values) => println!("  Positions (seq dim): {:?}", values),
                Err(e) => println!("  Evaluation error: {}", e),
            }
        }
        Err(e) => println!("  Compilation error: {}", e),
    }
}

fn tinybert_attention() {
    let mut compiler = LucidCompiler::new();
    // Simplified attention: average of current and previous value
    // This mimics how attention combines information from different positions
    let source = "(1 fby 2 fby 3 fby 4) + (0 fby 1 fby 2 fby 3)";

    match compiler.compile(source) {
        Ok(_) => {
            match compiler.eval(6) {
                Ok(values) => println!("  Attended values (curr + prev): {:?}", values),
                Err(e) => println!("  Evaluation error: {}", e),
            }
        }
        Err(e) => println!("  Compilation error: {}", e),
    }
}

fn tinybert_feedforward() {
    let mut compiler = LucidCompiler::new();
    // Feed-forward: linear transform + ReLU-like activation
    // hidden = x * 2 + 1, output = if hidden > 0 then hidden else 0
    let source = "if (1 fby 2 fby 3) * 2 + 1 > 0 then (1 fby 2 fby 3) * 2 + 1 else 0";

    match compiler.compile(source) {
        Ok(_) => {
            match compiler.eval(5) {
                Ok(values) => println!("  Feed-forward output: {:?}", values),
                Err(e) => println!("  Evaluation error: {}", e),
            }
        }
        Err(e) => println!("  Compilation error: {}", e),
    }
}

fn tinybert_residual() {
    let mut compiler = LucidCompiler::new();
    // Residual connection: output = input + transform(input)
    // Simulates skip connections in transformers
    let source = "(1 fby 2 fby 3) + ((1 fby 2 fby 3) * 2)";

    match compiler.compile(source) {
        Ok(_) => {
            match compiler.eval(5) {
                Ok(values) => println!("  Residual output (x + 2x = 3x): {:?}", values),
                Err(e) => println!("  Evaluation error: {}", e),
            }
        }
        Err(e) => println!("  Compilation error: {}", e),
    }
}

fn tinybert_layernorm() {
    let mut compiler = LucidCompiler::new();
    // Running sum for layer norm approximation
    // Accumulates values over time dimension
    let source = "1 fby.t 3 fby.t 6 fby.t 10";

    match compiler.compile(source) {
        Ok(_) => {
            match compiler.eval(6) {
                Ok(values) => println!("  Running sums (time dim): {:?}", values),
                Err(e) => println!("  Evaluation error: {}", e),
            }
        }
        Err(e) => println!("  Compilation error: {}", e),
    }
}

fn tinybert_multihead() {
    let mut compiler = LucidCompiler::new();
    // Multi-head attention: process input through multiple heads and combine
    // head1 = x * 1, head2 = x * 2, output = (head1 + head2) / 2
    let source = "((1 fby 2 fby 3) * 1 + (1 fby 2 fby 3) * 2) / 2";

    match compiler.compile(source) {
        Ok(_) => {
            match compiler.eval(5) {
                Ok(values) => println!("  Multi-head output: {:?}", values),
                Err(e) => println!("  Evaluation error: {}", e),
            }
        }
        Err(e) => println!("  Compilation error: {}", e),
    }
}

fn tinybert_full_layer() {
    let mut compiler = LucidCompiler::new();
    // Full TinyBERT-like layer with multi-dimensional annotations:
    // 1. Input embedding stream (.t dimension)
    // 2. Add positional encoding (.seq dimension)
    // 3. Self-attention (simplified)
    // 4. Feed-forward
    // 5. Residual connection
    //
    // Simplified as: ((emb + pos) + attention) * ff_weight + residual
    let source = r#"
        ((1 fby.t 2 fby.t 3 fby.t 4) + (0 fby.seq 1 fby.seq 2 fby.seq 3)) * 2 + (1 fby.t 2 fby.t 3 fby.t 4)
    "#;

    match compiler.compile(source) {
        Ok(_) => {
            println!("  Full layer compiled successfully with dimension annotations!");
            compiler.print_optimization_stats();

            match compiler.eval(6) {
                Ok(values) => println!("  Full layer output: {:?}", values),
                Err(e) => println!("  Evaluation error: {}", e),
            }
        }
        Err(e) => println!("  Compilation error: {}", e),
    }
}

fn tinybert_tensors() {
    let mut compiler = LucidCompiler::new();
    // Demonstrate tensor literals for weight matrices
    let source = "[1, 2, 3]";

    match compiler.compile(source) {
        Ok(_) => {
            match compiler.eval(3) {
                Ok(values) => println!("  Tensor literal: {:?}", values),
                Err(e) => println!("  Evaluation error: {}", e),
            }
        }
        Err(e) => println!("  Compilation error: {}", e),
    }
}

fn tinybert_batch() {
    let mut compiler = LucidCompiler::new();
    // Demonstrate batch dimension processing
    // Process multiple sequences in parallel
    let source = "1 fby.batch 2 fby.batch 3";

    match compiler.compile(source) {
        Ok(_) => {
            match compiler.eval(5) {
                Ok(values) => println!("  Batch processing: {:?}", values),
                Err(e) => println!("  Evaluation error: {}", e),
            }
        }
        Err(e) => println!("  Compilation error: {}", e),
    }
}

// Additional test to demonstrate the full power
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_stream_processing() {
        let mut compiler = LucidCompiler::new();
        
        // Create a stream and process it
        let source = "1 + 2";
        compiler.compile(source).unwrap();
        
        let result = compiler.eval(5).unwrap();
        assert_eq!(result.len(), 5);
    }
    
    #[test]
    fn test_temporal_operators() {
        let mut compiler = LucidCompiler::new();
        
        // Test fby operator
        let source = "0 fby 1";
        compiler.compile(source).unwrap();
        
        let result = compiler.eval(3).unwrap();
        println!("Temporal test result: {:?}", result);
    }
}
