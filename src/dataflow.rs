use std::collections::{HashMap, HashSet};
use std::fmt;

pub type NodeId = usize;

#[derive(Debug, Clone)]
pub struct DataflowGraph {
    pub nodes: HashMap<NodeId, Node>,
    pub edges: HashMap<NodeId, Vec<Edge>>,
    pub entry_point: NodeId,
    next_id: NodeId,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub to: NodeId,
    pub delay: usize, // 0 for immediate, >0 for temporal delay
}

#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    // Constants
    Constant(Value),

    // Variables
    Input(String),

    // Operations
    BinOp {
        op: crate::ast::BinOp,
        left: NodeId,
        right: NodeId,
    },
    UnOp {
        op: crate::ast::UnOp,
        operand: NodeId,
    },

    // Temporal with optional dimension
    Delay {
        input: NodeId,
        amount: usize,
        init: Value,
        dim: Option<String>, // Dimension name for multi-dimensional delays
    },

    // Control flow
    Select {
        cond: NodeId,
        then_val: NodeId,
        else_val: NodeId,
    },

    // Functions
    Lambda {
        params: Vec<String>,
        body: NodeId,
    },
    Apply {
        func: NodeId,
        args: Vec<NodeId>,
    },

    // Memoization node for CSE
    Memo {
        expr: NodeId,
    },

    // Tensor operations
    TensorLiteral(Vec<NodeId>),
    Index {
        tensor: NodeId,
        indices: Vec<NodeId>,
    },
    Dot {
        left: NodeId,
        right: NodeId,
    },
    Reduce {
        op: crate::ast::ReduceOp,
        dim: Option<String>,
        input: NodeId,
    },
    Reshape {
        input: NodeId,
        shape: Vec<NodeId>,
    },
    Transpose {
        input: NodeId,
        dim1: String,
        dim2: String,
    },

    // ML operations with optional dimension annotation
    MatMul {
        dim: Option<String>,
        left: NodeId,
        right: NodeId,
    },
    Relu {
        input: NodeId,
    },
    Softmax {
        dim: Option<String>,
        input: NodeId,
    },
    LayerNorm {
        dim: Option<String>,
        input: NodeId,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Stream(Vec<Value>), // For evaluation
    Tensor(Vec<Value>, Vec<usize>), // Data and shape for multi-dimensional values
}

impl Value {
    /// Create a 1D tensor from integers
    pub fn tensor_1d(values: Vec<i64>) -> Self {
        let len = values.len();
        Value::Tensor(values.into_iter().map(Value::Int).collect(), vec![len])
    }

    /// Create a 2D tensor (matrix)
    pub fn tensor_2d(values: Vec<Vec<i64>>) -> Self {
        let rows = values.len();
        let cols = if rows > 0 { values[0].len() } else { 0 };
        let flat: Vec<Value> = values.into_iter().flatten().map(Value::Int).collect();
        Value::Tensor(flat, vec![rows, cols])
    }

    /// Get tensor shape
    pub fn shape(&self) -> Vec<usize> {
        match self {
            Value::Tensor(_, shape) => shape.clone(),
            Value::Stream(vals) => vec![vals.len()],
            _ => vec![], // Scalar
        }
    }

    /// Check if value is a scalar
    pub fn is_scalar(&self) -> bool {
        matches!(self, Value::Int(_) | Value::Float(_) | Value::Bool(_))
    }
}

impl DataflowGraph {
    pub fn new() -> Self {
        DataflowGraph {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            entry_point: 0,
            next_id: 0,
        }
    }
    
    pub fn add_node(&mut self, node: Node) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, node);
        self.edges.insert(id, Vec::new());
        id
    }
    
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, delay: usize) {
        self.edges.entry(from).or_insert_with(Vec::new).push(Edge { to, delay });
    }
    
    pub fn get_dependencies(&self, node_id: NodeId) -> Vec<NodeId> {
        match self.nodes.get(&node_id) {
            Some(Node::BinOp { left, right, .. }) => vec![*left, *right],
            Some(Node::UnOp { operand, .. }) => vec![*operand],
            Some(Node::Delay { input, .. }) => vec![*input],
            Some(Node::Select { cond, then_val, else_val }) => {
                vec![*cond, *then_val, *else_val]
            }
            Some(Node::Apply { func, args }) => {
                let mut deps = vec![*func];
                deps.extend(args.iter());
                deps
            }
            Some(Node::Memo { expr }) => vec![*expr],
            Some(Node::TensorLiteral(elements)) => elements.clone(),
            Some(Node::Index { tensor, indices }) => {
                let mut deps = vec![*tensor];
                deps.extend(indices.iter());
                deps
            }
            Some(Node::Dot { left, right }) => vec![*left, *right],
            Some(Node::Reduce { input, .. }) => vec![*input],
            Some(Node::Reshape { input, shape }) => {
                let mut deps = vec![*input];
                deps.extend(shape.iter());
                deps
            }
            Some(Node::Transpose { input, .. }) => vec![*input],
            _ => Vec::new(),
        }
    }
    
    pub fn get_delay(&self, node_id: NodeId) -> usize {
        match self.nodes.get(&node_id) {
            Some(Node::Delay { amount, .. }) => *amount,
            _ => 0,
        }
    }
    
    // Find all nodes reachable from a given node
    pub fn reachable(&self, start: NodeId) -> HashSet<NodeId> {
        let mut visited = HashSet::new();
        let mut stack = vec![start];
        
        while let Some(node) = stack.pop() {
            if visited.insert(node) {
                stack.extend(self.get_dependencies(node));
            }
        }
        
        visited
    }
    
    // Topological sort for evaluation order
    pub fn topological_sort(&self, nodes: &HashSet<NodeId>) -> Vec<NodeId> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adj_list: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        
        for &node in nodes {
            in_degree.entry(node).or_insert(0);
            for dep in self.get_dependencies(node) {
                if nodes.contains(&dep) {
                    *in_degree.entry(node).or_insert(0) += 1;
                    adj_list.entry(dep).or_insert_with(Vec::new).push(node);
                }
            }
        }
        
        let mut queue: Vec<NodeId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();
        
        let mut result = Vec::new();
        
        while let Some(node) = queue.pop() {
            result.push(node);
            
            if let Some(neighbors) = adj_list.get(&node) {
                for &neighbor in neighbors {
                    if let Some(degree) = in_degree.get_mut(&neighbor) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push(neighbor);
                        }
                    }
                }
            }
        }
        
        result
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(x) => write!(f, "{}", x),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Stream(vals) => {
                write!(f, "[")?;
                for (i, val) in vals.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", val)?;
                }
                write!(f, "]")
            }
            Value::Tensor(vals, shape) => {
                write!(f, "Tensor<{:?}>[", shape)?;
                for (i, val) in vals.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    if i >= 10 {
                        write!(f, "...")?;
                        break;
                    }
                    write!(f, "{}", val)?;
                }
                write!(f, "]")
            }
        }
    }
}

impl Default for DataflowGraph {
    fn default() -> Self {
        Self::new()
    }
}
