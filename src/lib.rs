// Core compiler modules
pub mod lexer;
pub mod ast;
pub mod parser;
pub mod dataflow;
pub mod graph_builder;
pub mod demand_analysis;
pub mod cse;
pub mod loop_fusion;
pub mod evaluator;
pub mod optimizer;

// ML training modules
pub mod tensor;
pub mod autodiff;
pub mod loss;
pub mod optimizer_ml;
pub mod data;
pub mod trainer;

use ast::Program;
use dataflow::DataflowGraph;
use evaluator::{DemandEvaluator, EvalError};
use optimizer::Optimizer;

// Re-export commonly used types for convenience
pub use tensor::TensorValue;
pub use autodiff::GradientTape;
pub use loss::LossType;
pub use optimizer_ml::{MLOptimizer, OptimizerConfig, LRScheduler};
pub use data::{Dataset, DataLoader, Sample};
pub use trainer::{Trainer, TrainingConfig, TrainingMetrics, SimpleModel};

/// Main compiler pipeline
pub struct LucidCompiler {
    program: Option<Program>,
    graph: Option<DataflowGraph>,
    optimizer: Optimizer,
}

impl LucidCompiler {
    pub fn new() -> Self {
        LucidCompiler {
            program: None,
            graph: None,
            optimizer: Optimizer::new(),
        }
    }

    /// Compile from source code
    pub fn compile(&mut self, source: &str) -> Result<(), CompileError> {
        // Lex
        println!("Lexing...");
        let tokens = lexer::tokenize(source);

        // Parse
        println!("Parsing...");
        let program = parser::parse(tokens)
            .map_err(|e| CompileError::ParseError(format!("{:?}", e)))?;

        // Build dataflow graph
        println!("Building dataflow graph...");
        let mut graph = graph_builder::ast_to_dataflow(&program);

        // Optimize
        println!("\nOptimizing...");
        self.optimizer.optimize(&mut graph);

        self.program = Some(program);
        self.graph = Some(graph);

        Ok(())
    }

    /// Evaluate the compiled program for a number of time steps
    pub fn eval(&self, steps: usize) -> Result<Vec<dataflow::Value>, EvalError> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| EvalError::NotImplemented("Program not compiled".to_string()))?;

        let mut evaluator = DemandEvaluator::new(graph.clone());
        evaluator.eval_stream(graph.entry_point, steps)
    }

    /// Get the optimized dataflow graph
    pub fn get_graph(&self) -> Option<&DataflowGraph> {
        self.graph.as_ref()
    }

    /// Get optimization statistics
    pub fn print_optimization_stats(&self) {
        if let Some(ref graph) = self.graph {
            self.optimizer.print_stats(graph);
        }
    }
}

impl Default for LucidCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub enum CompileError {
    LexError(String),
    ParseError(String),
    GraphBuildError(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            CompileError::LexError(msg) => write!(f, "Lexer error: {}", msg),
            CompileError::ParseError(msg) => write!(f, "Parser error: {}", msg),
            CompileError::GraphBuildError(msg) => write!(f, "Graph build error: {}", msg),
        }
    }
}

impl std::error::Error for CompileError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_compile() {
        let mut compiler = LucidCompiler::new();
        let result = compiler.compile("1 + 2");
        assert!(result.is_ok());
    }

    #[test]
    fn test_fibonacci() {
        let mut compiler = LucidCompiler::new();
        // Simplified fibonacci without recursive where clause
        let source = "0 fby 1";
        let result = compiler.compile(source);
        assert!(result.is_ok());

        if result.is_ok() {
            let values = compiler.eval(5).unwrap();
            println!("Fibonacci sequence: {:?}", values);
        }
    }
}
