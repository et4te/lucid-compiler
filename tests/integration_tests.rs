#[cfg(test)]
mod tests {
    use lucid_compiler::*;
    use lucid_compiler::dataflow::*;
    use lucid_compiler::ast::*;
    
    #[test]
    fn test_lexer_basic() {
        use lucid_compiler::lexer::*;
        
        let tokens = tokenize("fib = 0 fby 1");
        assert!(tokens.iter().any(|t| matches!(t, Token::Fby)));
        assert!(tokens.iter().any(|t| matches!(t, Token::Eq)));
    }
    
    #[test]
    fn test_lexer_temporal_operators() {
        use lucid_compiler::lexer::*;
        
        let input = "next prev first fby";
        let tokens = tokenize(input);
        
        assert!(tokens.contains(&Token::Next));
        assert!(tokens.contains(&Token::Prev));
        assert!(tokens.contains(&Token::First));
        assert!(tokens.contains(&Token::Fby));
    }
    
    #[test]
    fn test_parser_arithmetic() {
        use lucid_compiler::lexer::*;
        use lucid_compiler::parser::*;
        
        let tokens = tokenize("1 + 2 * 3");
        let result = parse(tokens);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_parser_fby() {
        use lucid_compiler::lexer::*;
        use lucid_compiler::parser::*;
        
        let tokens = tokenize("0 fby 1 fby 2");
        let result = parse(tokens);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_graph_builder_constant() {
        use lucid_compiler::graph_builder::*;
        
        let program = Program {
            definitions: vec![],
            params: vec![],
            train_config: None,
            main_expr: Expr::Int(42),
        };
        
        let graph = ast_to_dataflow(&program);
        assert!(!graph.nodes.is_empty());
    }
    
    #[test]
    fn test_graph_builder_fby() {
        use lucid_compiler::graph_builder::*;
        
        let expr = Expr::fby(Expr::int(0), Expr::int(1));
        let program = Program {
            definitions: vec![],
            params: vec![],
            train_config: None,
            main_expr: expr,
        };
        
        let graph = ast_to_dataflow(&program);
        
        // Should have at least: 2 constants + 1 delay
        assert!(graph.nodes.len() >= 3);
    }
    
    #[test]
    fn test_demand_analysis() {
        use lucid_compiler::demand_analysis::*;
        
        let mut graph = DataflowGraph::new();
        
        let const1 = graph.add_node(Node::Constant(Value::Int(1)));
        let const2 = graph.add_node(Node::Constant(Value::Int(2)));
        let sum = graph.add_node(Node::BinOp {
            op: BinOp::Add,
            left: const1,
            right: const2,
        });
        
        // const2 is not reachable from const1
        graph.entry_point = const1;
        
        let analysis = DemandAnalysis::analyze(&graph, const1);
        
        // Only const1 should be demanded
        assert!(analysis.demanded.contains(&const1));
        assert!(!analysis.demanded.contains(&sum));
    }
    
    #[test]
    fn test_cse_elimination() {
        use lucid_compiler::cse::CSE;

        let mut graph = DataflowGraph::new();

        let a = graph.add_node(Node::Constant(Value::Int(5)));
        let b = graph.add_node(Node::Constant(Value::Int(3)));

        // Create two identical additions
        let _add1 = graph.add_node(Node::BinOp {
            op: BinOp::Add,
            left: a,
            right: b,
        });
        let add2 = graph.add_node(Node::BinOp {
            op: BinOp::Add,
            left: a,
            right: b,
        });

        // Create a result that uses BOTH adds to make them reachable
        let result = graph.add_node(Node::BinOp {
            op: BinOp::Mul,
            left: _add1,
            right: add2,
        });

        graph.entry_point = result;

        let cse = CSE::new();
        let mapping = cse.optimize(&mut graph);

        // One of the additions should be mapped to the other
        assert!(!mapping.is_empty(), "CSE should deduplicate identical additions");

        // After CSE, the result node should reference the same add node for both operands
        if let Some(Node::BinOp { left, right, .. }) = graph.nodes.get(&result) {
            assert_eq!(left, right, "After CSE, both operands should reference the same node");
        }
    }
    
    #[test]
    fn test_evaluator_constant() {
        use lucid_compiler::evaluator::*;
        
        let mut graph = DataflowGraph::new();
        let node = graph.add_node(Node::Constant(Value::Int(42)));
        graph.entry_point = node;
        
        let mut eval = DemandEvaluator::new(graph);
        let result = eval.eval(node, 0).unwrap();
        
        assert_eq!(result, Value::Int(42));
    }
    
    #[test]
    fn test_evaluator_binop() {
        use lucid_compiler::evaluator::*;
        
        let mut graph = DataflowGraph::new();
        
        let a = graph.add_node(Node::Constant(Value::Int(10)));
        let b = graph.add_node(Node::Constant(Value::Int(5)));
        let sum = graph.add_node(Node::BinOp {
            op: BinOp::Add,
            left: a,
            right: b,
        });
        
        graph.entry_point = sum;
        
        let mut eval = DemandEvaluator::new(graph);
        let result = eval.eval(sum, 0).unwrap();
        
        assert_eq!(result, Value::Int(15));
    }
    
    #[test]
    fn test_evaluator_delay() {
        use lucid_compiler::evaluator::*;
        
        let mut graph = DataflowGraph::new();
        
        let input = graph.add_node(Node::Constant(Value::Int(10)));
        let delayed = graph.add_node(Node::Delay {
            input,
            amount: 2,
            init: Value::Int(0),
            dim: None,
        });
        
        graph.entry_point = delayed;
        
        let mut eval = DemandEvaluator::new(graph);
        
        // First two steps should return init value
        assert_eq!(eval.eval(delayed, 0).unwrap(), Value::Int(0));
        assert_eq!(eval.eval(delayed, 1).unwrap(), Value::Int(0));
        
        // After delay, should get actual value
        assert_eq!(eval.eval(delayed, 2).unwrap(), Value::Int(10));
    }
    
    #[test]
    fn test_evaluator_memoization() {
        use lucid_compiler::evaluator::*;
        
        let mut graph = DataflowGraph::new();
        
        let a = graph.add_node(Node::Constant(Value::Int(5)));
        let b = graph.add_node(Node::Constant(Value::Int(3)));
        let sum = graph.add_node(Node::BinOp {
            op: BinOp::Add,
            left: a,
            right: b,
        });
        
        let mut eval = DemandEvaluator::new(graph);
        
        // Evaluate twice at same time
        eval.eval(sum, 0).unwrap();
        eval.eval(sum, 0).unwrap();
        
        let (cache_size, _) = eval.cache_stats();
        
        // Should be cached
        assert!(cache_size > 0);
    }
    
    #[test]
    fn test_loop_fusion() {
        use lucid_compiler::loop_fusion::LoopFusion;
        
        let mut graph = DataflowGraph::new();
        
        let a = graph.add_node(Node::Constant(Value::Int(1)));
        let _delay_a = graph.add_node(Node::Delay {
            input: a,
            amount: 1,
            init: Value::Int(0),
            dim: None,
        });
        
        let fusion = LoopFusion::new().analyze(&graph);
        
        // Should create at least one fusion group
        assert!(!fusion.get_schedule().is_empty());
    }
    
    #[test]
    fn test_buffer_minimization() {
        use lucid_compiler::loop_fusion::{LoopFusion, BufferMinimization};
        
        let mut graph = DataflowGraph::new();
        
        let input = graph.add_node(Node::Constant(Value::Int(0)));
        let delay1 = graph.add_node(Node::Delay {
            input,
            amount: 1,
            init: Value::Int(0),
            dim: None,
        });
        let delay2 = graph.add_node(Node::Delay {
            input: delay1,
            amount: 2,
            init: Value::Int(0),
            dim: None,
        });
        
        graph.entry_point = delay2;
        
        let fusion = LoopFusion::new().analyze(&graph);
        let buffer_min = BufferMinimization::new().analyze(&graph, &fusion);
        
        // Should calculate buffer requirements
        assert!(buffer_min.get_buffer_size(input) >= 1);
    }
    
    #[test]
    fn test_full_compilation() {
        let mut compiler = LucidCompiler::new();
        
        let result = compiler.compile("(5 + 3) * 2");
        assert!(result.is_ok());
        
        let values = compiler.eval(1).unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(values[0], Value::Int(16));
    }
    
    #[test]
    fn test_fby_compilation() {
        let mut compiler = LucidCompiler::new();
        
        let result = compiler.compile("0 fby 1");
        assert!(result.is_ok());
        
        let values = compiler.eval(5).unwrap();
        assert_eq!(values.len(), 5);
        assert_eq!(values[0], Value::Int(0));
        assert_eq!(values[1], Value::Int(1));
    }
    
    #[test]
    fn test_optimization_pipeline() {
        let mut compiler = LucidCompiler::new();
        
        // Expression with redundancy
        let result = compiler.compile("(2 + 3) + (2 + 3)");
        assert!(result.is_ok());
        
        // Should optimize away duplicate computation
        let values = compiler.eval(1).unwrap();
        assert_eq!(values[0], Value::Int(10));
    }
    
    #[test]
    fn test_complex_expression() {
        let mut compiler = LucidCompiler::new();

        let result = compiler.compile("if 5 > 3 then 10 else 20");
        assert!(result.is_ok());

        let values = compiler.eval(1).unwrap();
        assert_eq!(values[0], Value::Int(10));
    }

    // ============================================
    // TinyBERT-Inspired Tests
    // ============================================

    #[test]
    fn test_tinybert_token_embedding() {
        let mut compiler = LucidCompiler::new();

        // Token embedding stream: simulates sequence of token IDs
        let result = compiler.compile("1 fby 2 fby 3 fby 4 fby 5");
        assert!(result.is_ok());

        let values = compiler.eval(7).unwrap();
        assert_eq!(values[0], Value::Int(1));
        assert_eq!(values[1], Value::Int(2));
        assert_eq!(values[2], Value::Int(3));
        assert_eq!(values[3], Value::Int(4));
        assert_eq!(values[4], Value::Int(5));
        // After the sequence, it should repeat the last value
        assert_eq!(values[5], Value::Int(5));
    }

    #[test]
    fn test_tinybert_positional_encoding() {
        let mut compiler = LucidCompiler::new();

        // Positional encoding: 0, 1, 2, 3, ...
        let result = compiler.compile("0 fby 1 fby 2 fby 3 fby 4");
        assert!(result.is_ok());

        let values = compiler.eval(6).unwrap();
        assert_eq!(values[0], Value::Int(0));
        assert_eq!(values[1], Value::Int(1));
        assert_eq!(values[2], Value::Int(2));
        assert_eq!(values[3], Value::Int(3));
        assert_eq!(values[4], Value::Int(4));
    }

    #[test]
    fn test_tinybert_attention_combine() {
        let mut compiler = LucidCompiler::new();

        // Attention: combine current and previous values
        // (current + previous) simulates attention aggregation
        let result = compiler.compile("(1 fby 2 fby 3) + (0 fby 1 fby 2)");
        assert!(result.is_ok());

        let values = compiler.eval(5).unwrap();
        // t=0: 1 + 0 = 1
        // t=1: 2 + 1 = 3
        // t=2: 3 + 2 = 5
        assert_eq!(values[0], Value::Int(1));
        assert_eq!(values[1], Value::Int(3));
        assert_eq!(values[2], Value::Int(5));
    }

    #[test]
    fn test_tinybert_feedforward_relu() {
        let mut compiler = LucidCompiler::new();

        // Feed-forward with ReLU: if x > 0 then x else 0
        let result = compiler.compile("if 5 > 0 then 5 * 2 else 0");
        assert!(result.is_ok());

        let values = compiler.eval(1).unwrap();
        assert_eq!(values[0], Value::Int(10));
    }

    #[test]
    fn test_tinybert_feedforward_negative() {
        let mut compiler = LucidCompiler::new();

        // ReLU should return 0 for negative values
        // Since we can't have negative literals directly, we compute one
        let result = compiler.compile("if 0 > 5 then 10 else 0");
        assert!(result.is_ok());

        let values = compiler.eval(1).unwrap();
        assert_eq!(values[0], Value::Int(0));
    }

    #[test]
    fn test_tinybert_residual_connection() {
        let mut compiler = LucidCompiler::new();

        // Residual: x + transform(x) where transform = x * 2
        // Result should be x + 2x = 3x
        let result = compiler.compile("(1 fby 2 fby 3) + ((1 fby 2 fby 3) * 2)");
        assert!(result.is_ok());

        let values = compiler.eval(4).unwrap();
        // t=0: 1 + 2 = 3
        // t=1: 2 + 4 = 6
        // t=2: 3 + 6 = 9
        assert_eq!(values[0], Value::Int(3));
        assert_eq!(values[1], Value::Int(6));
        assert_eq!(values[2], Value::Int(9));
    }

    #[test]
    fn test_tinybert_multihead_attention() {
        let mut compiler = LucidCompiler::new();

        // Multi-head: average of head1 (x*1) and head2 (x*2)
        // Result = (x*1 + x*2) / 2 = 3x/2
        // For integer division: (2*1 + 2*2) / 2 = 6/2 = 3
        let result = compiler.compile("((2 fby 4 fby 6) * 1 + (2 fby 4 fby 6) * 2) / 2");
        assert!(result.is_ok());

        let values = compiler.eval(4).unwrap();
        // t=0: (2 + 4) / 2 = 3
        // t=1: (4 + 8) / 2 = 6
        // t=2: (6 + 12) / 2 = 9
        assert_eq!(values[0], Value::Int(3));
        assert_eq!(values[1], Value::Int(6));
        assert_eq!(values[2], Value::Int(9));
    }

    #[test]
    fn test_tinybert_layer_composition() {
        let mut compiler = LucidCompiler::new();

        // Compose multiple transformer operations:
        // 1. Input embedding + positional encoding
        // 2. Multiply by weight (simulating attention/FF)
        // 3. Add residual
        let source = "((1 fby 2 fby 3) + (0 fby 1 fby 2)) * 2 + (1 fby 2 fby 3)";
        let result = compiler.compile(source);
        assert!(result.is_ok());

        let values = compiler.eval(4).unwrap();
        // t=0: (1+0)*2 + 1 = 2 + 1 = 3
        // t=1: (2+1)*2 + 2 = 6 + 2 = 8
        // t=2: (3+2)*2 + 3 = 10 + 3 = 13
        assert_eq!(values[0], Value::Int(3));
        assert_eq!(values[1], Value::Int(8));
        assert_eq!(values[2], Value::Int(13));
    }

    #[test]
    fn test_tinybert_running_statistics() {
        let mut compiler = LucidCompiler::new();

        // Running sum for layer normalization approximation
        // sum = 1, 3, 6, 10, ... (triangular numbers)
        let result = compiler.compile("1 fby 3 fby 6 fby 10 fby 15");
        assert!(result.is_ok());

        let values = compiler.eval(6).unwrap();
        assert_eq!(values[0], Value::Int(1));
        assert_eq!(values[1], Value::Int(3));
        assert_eq!(values[2], Value::Int(6));
        assert_eq!(values[3], Value::Int(10));
        assert_eq!(values[4], Value::Int(15));
    }

    #[test]
    fn test_tinybert_cse_optimization() {
        let mut compiler = LucidCompiler::new();

        // Common subexpression: (1 fby 2) appears twice
        // CSE should optimize this
        let result = compiler.compile("(1 fby 2 fby 3) + (1 fby 2 fby 3)");
        assert!(result.is_ok());

        let values = compiler.eval(4).unwrap();
        // t=0: 1 + 1 = 2
        // t=1: 2 + 2 = 4
        // t=2: 3 + 3 = 6
        assert_eq!(values[0], Value::Int(2));
        assert_eq!(values[1], Value::Int(4));
        assert_eq!(values[2], Value::Int(6));
    }

    #[test]
    fn test_tinybert_full_forward_pass() {
        let mut compiler = LucidCompiler::new();

        // Full forward pass simulation:
        // Layer 1: embedding + positional
        // Layer 2: attention (combine with previous)
        // Layer 3: feed-forward (multiply by 2)
        // Layer 4: residual connection
        //
        // Simplified: ((emb + pos + prev_attended) * 2 + emb)
        let source = r#"
            if ((1 fby 2 fby 3) + (0 fby 1 fby 2)) * 2 > 0
            then ((1 fby 2 fby 3) + (0 fby 1 fby 2)) * 2
            else 0
        "#;
        let result = compiler.compile(source);
        assert!(result.is_ok(), "Full forward pass should compile");

        let values = compiler.eval(4).unwrap();
        // All values should be positive (ReLU passes through)
        // t=0: (1+0)*2 = 2
        // t=1: (2+1)*2 = 6
        // t=2: (3+2)*2 = 10
        assert_eq!(values[0], Value::Int(2));
        assert_eq!(values[1], Value::Int(6));
        assert_eq!(values[2], Value::Int(10));
    }
}
