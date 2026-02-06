use crate::dataflow::*;
use std::collections::HashMap;

/// Demand-driven evaluator with memoization for lazy stream evaluation
pub struct DemandEvaluator {
    graph: DataflowGraph,
    // Cache: (node_id, time_step) -> Value
    cache: HashMap<(NodeId, usize), Value>,
    // Current environment for lambda evaluation
    env: HashMap<String, NodeId>,
}

impl DemandEvaluator {
    pub fn new(graph: DataflowGraph) -> Self {
        DemandEvaluator {
            graph,
            cache: HashMap::new(),
            env: HashMap::new(),
        }
    }
    
    /// Evaluate a node at a specific time step
    pub fn eval(&mut self, node_id: NodeId, time: usize) -> Result<Value, EvalError> {
        // Check cache first
        if let Some(cached) = self.cache.get(&(node_id, time)) {
            return Ok(cached.clone());
        }
        
        let result = self.eval_node(node_id, time)?;
        
        // Memoize result
        self.cache.insert((node_id, time), result.clone());
        
        Ok(result)
    }
    
    fn eval_node(&mut self, node_id: NodeId, time: usize) -> Result<Value, EvalError> {
        let node = self.graph.nodes.get(&node_id)
            .ok_or(EvalError::NodeNotFound(node_id))?
            .clone();
        
        match node {
            Node::Constant(val) => Ok(val),
            
            Node::Input(name) => {
                // For inputs, we'd need to get values from somewhere
                // For now, return a default or error
                Err(EvalError::UndefinedInput(name))
            }
            
            Node::BinOp { op, left, right } => {
                let left_val = self.eval(left, time)?;
                let right_val = self.eval(right, time)?;
                self.apply_binop(op, left_val, right_val)
            }
            
            Node::UnOp { op, operand } => {
                let val = self.eval(operand, time)?;
                self.apply_unop(op, val)
            }
            
            Node::Delay { input, amount, init, .. } => {
                if time < amount {
                    // Return initial value for early time steps
                    Ok(init)
                } else {
                    // Get value from input at earlier time
                    self.eval(input, time - amount)
                }
            }
            
            Node::Select { cond, then_val, else_val } => {
                let cond_result = self.eval(cond, time)?;
                match cond_result {
                    Value::Bool(true) => self.eval(then_val, time),
                    Value::Bool(false) => self.eval(else_val, time),
                    _ => Err(EvalError::TypeError("Expected boolean in condition".to_string())),
                }
            }
            
            Node::Lambda { params, body } => {
                // Return a closure representation
                // In a real implementation, we'd need a proper closure value
                Ok(Value::Int(body as i64)) // Placeholder
            }
            
            Node::Apply { func, args } => {
                // Evaluate function
                let func_node = match &self.graph.nodes.get(&func) {
                    Some(Node::Lambda { params, body }) => {
                        if params.len() != args.len() {
                            return Err(EvalError::ArityMismatch {
                                expected: params.len(),
                                got: args.len(),
                            });
                        }
                        
                        // Create new environment with argument bindings
                        let saved_env = self.env.clone();
                        
                        for (param, arg_id) in params.iter().zip(args.iter()) {
                            self.env.insert(param.clone(), *arg_id);
                        }
                        
                        let result = self.eval(*body, time);
                        
                        self.env = saved_env;
                        
                        return result;
                    }
                    _ => return Err(EvalError::NotAFunction),
                };
                
                Err(EvalError::NotImplemented("Complex function application".to_string()))
            }
            
            Node::Memo { expr } => {
                // Memoization is handled by the cache, just evaluate
                self.eval(expr, time)
            }

            // Tensor operations
            Node::TensorLiteral(elements) => {
                let mut vals = Vec::new();
                for &elem in &elements {
                    vals.push(self.eval(elem, time)?);
                }
                let len = vals.len();
                Ok(Value::Tensor(vals, vec![len]))
            }

            Node::Index { tensor, indices } => {
                let tensor_val = self.eval(tensor, time)?;
                let mut idx_vals = Vec::new();
                for &idx in &indices {
                    idx_vals.push(self.eval(idx, time)?);
                }

                match tensor_val {
                    Value::Tensor(data, shape) => {
                        if indices.len() == 1 {
                            if let Value::Int(i) = idx_vals[0] {
                                let i = i as usize;
                                if i < data.len() {
                                    Ok(data[i].clone())
                                } else {
                                    Err(EvalError::IndexOutOfBounds(i, data.len()))
                                }
                            } else {
                                Err(EvalError::TypeError("Index must be integer".to_string()))
                            }
                        } else {
                            Err(EvalError::NotImplemented("Multi-dimensional indexing".to_string()))
                        }
                    }
                    _ => Err(EvalError::TypeError("Cannot index non-tensor".to_string())),
                }
            }

            Node::Dot { left, right } => {
                let left_val = self.eval(left, time)?;
                let right_val = self.eval(right, time)?;

                match (left_val, right_val) {
                    (Value::Tensor(a, _), Value::Tensor(b, _)) => {
                        // Simple dot product for 1D tensors
                        if a.len() != b.len() {
                            return Err(EvalError::TypeError("Tensor size mismatch for dot product".to_string()));
                        }
                        let mut sum = 0i64;
                        for (av, bv) in a.iter().zip(b.iter()) {
                            match (av, bv) {
                                (Value::Int(x), Value::Int(y)) => sum += x * y,
                                _ => return Err(EvalError::TypeError("Dot product requires numeric tensors".to_string())),
                            }
                        }
                        Ok(Value::Int(sum))
                    }
                    _ => Err(EvalError::TypeError("Dot product requires tensors".to_string())),
                }
            }

            Node::Reduce { op, input, .. } => {
                let input_val = self.eval(input, time)?;
                match input_val {
                    Value::Tensor(data, _) => {
                        use crate::ast::ReduceOp::*;
                        let result = match op {
                            Sum => {
                                let mut sum = 0i64;
                                for v in &data {
                                    if let Value::Int(x) = v {
                                        sum += x;
                                    }
                                }
                                Value::Int(sum)
                            }
                            Mean => {
                                let mut sum = 0i64;
                                for v in &data {
                                    if let Value::Int(x) = v {
                                        sum += x;
                                    }
                                }
                                Value::Int(sum / data.len() as i64)
                            }
                            Max => {
                                let mut max = i64::MIN;
                                for v in &data {
                                    if let Value::Int(x) = v {
                                        max = max.max(*x);
                                    }
                                }
                                Value::Int(max)
                            }
                            Min => {
                                let mut min = i64::MAX;
                                for v in &data {
                                    if let Value::Int(x) = v {
                                        min = min.min(*x);
                                    }
                                }
                                Value::Int(min)
                            }
                            Prod => {
                                let mut prod = 1i64;
                                for v in &data {
                                    if let Value::Int(x) = v {
                                        prod *= x;
                                    }
                                }
                                Value::Int(prod)
                            }
                        };
                        Ok(result)
                    }
                    _ => Err(EvalError::TypeError("Reduce requires tensor".to_string())),
                }
            }

            Node::Reshape { input, .. } => {
                // For now, just return the input unchanged
                self.eval(input, time)
            }

            Node::Transpose { input, .. } => {
                // For now, just return the input unchanged (1D transpose is identity)
                self.eval(input, time)
            }

            Node::MatMul { left, right, .. } => {
                // Matrix multiplication - for now return left side
                // Full implementation would do actual matmul on tensors
                let _left_val = self.eval(left, time)?;
                let _right_val = self.eval(right, time)?;
                // Placeholder: return a simple result
                Ok(Value::Float(0.0))
            }

            Node::Relu { input } => {
                match self.eval(input, time)? {
                    Value::Float(x) => Ok(Value::Float(if x > 0.0 { x } else { 0.0 })),
                    Value::Int(x) => Ok(Value::Int(if x > 0 { x } else { 0 })),
                    other => Ok(other),
                }
            }

            Node::Softmax { input, .. } => {
                // Softmax - for now just pass through
                self.eval(input, time)
            }

            Node::LayerNorm { input, .. } => {
                // LayerNorm - for now just pass through
                self.eval(input, time)
            }
        }
    }
    
    fn apply_binop(&mut self, op: crate::ast::BinOp, left: Value, right: Value) -> Result<Value, EvalError> {
        use crate::ast::BinOp::*;
        
        match (op, left, right) {
            (Add, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
            (Sub, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
            (Mul, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
            (Div, Value::Int(a), Value::Int(b)) => {
                if b == 0 {
                    Err(EvalError::DivisionByZero)
                } else {
                    Ok(Value::Int(a / b))
                }
            }
            
            (Add, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
            (Sub, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
            (Mul, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
            (Div, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
            
            (Eq, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a == b)),
            (Neq, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a != b)),
            (Lt, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
            (Lte, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
            (Gt, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
            (Gte, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),
            
            (And, Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a && b)),
            (Or, Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a || b)),
            
            _ => Err(EvalError::TypeError(format!("Type mismatch in binary operation: {:?}", op))),
        }
    }
    
    fn apply_unop(&mut self, op: crate::ast::UnOp, val: Value) -> Result<Value, EvalError> {
        use crate::ast::UnOp::*;
        
        match (op, val) {
            (Neg, Value::Int(n)) => Ok(Value::Int(-n)),
            (Neg, Value::Float(f)) => Ok(Value::Float(-f)),
            (Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
            _ => Err(EvalError::TypeError("Type mismatch in unary operation".to_string())),
        }
    }
    
    /// Evaluate a stream for multiple time steps
    pub fn eval_stream(&mut self, node_id: NodeId, steps: usize) -> Result<Vec<Value>, EvalError> {
        let mut results = Vec::new();
        for t in 0..steps {
            results.push(self.eval(node_id, t)?);
        }
        Ok(results)
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        // Returns (cache_size, unique_nodes)
        let unique_nodes = self.cache.keys()
            .map(|(node_id, _)| node_id)
            .collect::<std::collections::HashSet<_>>()
            .len();
        (self.cache.len(), unique_nodes)
    }
}

#[derive(Debug, Clone)]
pub enum EvalError {
    NodeNotFound(NodeId),
    UndefinedInput(String),
    TypeError(String),
    DivisionByZero,
    NotAFunction,
    ArityMismatch { expected: usize, got: usize },
    NotImplemented(String),
    IndexOutOfBounds(usize, usize),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            EvalError::NodeNotFound(id) => write!(f, "Node {} not found", id),
            EvalError::UndefinedInput(name) => write!(f, "Undefined input: {}", name),
            EvalError::TypeError(msg) => write!(f, "Type error: {}", msg),
            EvalError::DivisionByZero => write!(f, "Division by zero"),
            EvalError::NotAFunction => write!(f, "Not a function"),
            EvalError::ArityMismatch { expected, got } => {
                write!(f, "Arity mismatch: expected {} arguments, got {}", expected, got)
            }
            EvalError::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
            EvalError::IndexOutOfBounds(idx, len) => {
                write!(f, "Index {} out of bounds for tensor of length {}", idx, len)
            }
        }
    }
}

impl std::error::Error for EvalError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::BinOp;
    
    #[test]
    fn test_constant_eval() {
        let mut graph = DataflowGraph::new();
        let node = graph.add_node(Node::Constant(Value::Int(42)));
        graph.entry_point = node;
        
        let mut eval = DemandEvaluator::new(graph);
        let result = eval.eval(node, 0).unwrap();
        
        assert_eq!(result, Value::Int(42));
    }
    
    #[test]
    fn test_delay_eval() {
        let mut graph = DataflowGraph::new();
        let input = graph.add_node(Node::Constant(Value::Int(5)));
        let delayed = graph.add_node(Node::Delay {
            input,
            amount: 2,
            init: Value::Int(0),
            dim: None,
        });
        graph.entry_point = delayed;
        
        let mut eval = DemandEvaluator::new(graph);
        
        assert_eq!(eval.eval(delayed, 0).unwrap(), Value::Int(0)); // Init value
        assert_eq!(eval.eval(delayed, 1).unwrap(), Value::Int(0)); // Still init
        assert_eq!(eval.eval(delayed, 2).unwrap(), Value::Int(5)); // Now the actual value
    }
    
    #[test]
    fn test_binop_eval() {
        let mut graph = DataflowGraph::new();
        let a = graph.add_node(Node::Constant(Value::Int(10)));
        let b = graph.add_node(Node::Constant(Value::Int(3)));
        let sum = graph.add_node(Node::BinOp {
            op: BinOp::Add,
            left: a,
            right: b,
        });
        
        let mut eval = DemandEvaluator::new(graph);
        assert_eq!(eval.eval(sum, 0).unwrap(), Value::Int(13));
    }
}
