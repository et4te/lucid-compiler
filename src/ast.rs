use std::fmt;

pub type Ident = String;

/// Dimension identifier for multi-dimensional temporal operations
/// Common dimensions: t (time), batch, seq (sequence), hidden, head
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dimension(pub String);

impl Dimension {
    pub fn new(name: impl Into<String>) -> Self {
        Dimension(name.into())
    }

    /// Default time dimension
    pub fn time() -> Self {
        Dimension("t".to_string())
    }

    /// Batch dimension for parallel processing
    pub fn batch() -> Self {
        Dimension("batch".to_string())
    }

    /// Sequence dimension for transformer inputs
    pub fn seq() -> Self {
        Dimension("seq".to_string())
    }

    /// Hidden dimension for embeddings
    pub fn hidden() -> Self {
        Dimension("hidden".to_string())
    }

    /// Attention head dimension
    pub fn head() -> Self {
        Dimension("head".to_string())
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Shape of a tensor, list of dimension sizes
#[derive(Debug, Clone, PartialEq)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn scalar() -> Self {
        Shape(vec![])
    }

    pub fn vector(n: usize) -> Self {
        Shape(vec![n])
    }

    pub fn matrix(rows: usize, cols: usize) -> Self {
        Shape(vec![rows, cols])
    }

    pub fn tensor(dims: Vec<usize>) -> Self {
        Shape(dims)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub definitions: Vec<Definition>,
    pub params: Vec<ParamDecl>,
    pub train_config: Option<TrainConfig>,
    pub main_expr: Expr,
}

/// Parameter declaration for trainable weights
#[derive(Debug, Clone, PartialEq)]
pub struct ParamDecl {
    pub name: Ident,
    pub shape: Vec<Expr>,
    pub initializer: Initializer,
}

/// Parameter initialization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Initializer {
    Xavier,
    He,
    Zeros,
    Ones,
}

/// Training configuration block
#[derive(Debug, Clone, PartialEq)]
pub struct TrainConfig {
    pub input: Ident,
    pub target: Ident,
    pub model: Expr,
    pub loss: LossFunction,
    pub optimizer: OptimizerType,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
}

/// Loss function types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossFunction {
    CrossEntropy,
    Mse,
}

/// Optimizer types
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizerType {
    pub kind: OptimizerKind,
    pub learning_rate: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerKind {
    Adam,
    Sgd,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Definition {
    Function {
        name: Ident,
        params: Vec<Ident>,
        body: Expr,
    },
    Equation {
        name: Ident,
        expr: Expr,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // Literals
    Int(i64),
    Float(f64),
    Bool(bool),
    Var(Ident),

    // Tensor literal: [1, 2, 3] or [[1, 2], [3, 4]]
    Tensor(Vec<Expr>),

    // Temporal operators with optional dimension annotation
    // fby.t means "followed by in time dimension"
    // fby.seq means "followed by in sequence dimension"
    Fby {
        dim: Option<Dimension>,
        init: Box<Expr>,
        next: Box<Expr>,
    },
    Next {
        dim: Option<Dimension>,
        expr: Box<Expr>,
    },
    Prev {
        dim: Option<Dimension>,
        expr: Box<Expr>,
    },
    First {
        dim: Option<Dimension>,
        expr: Box<Expr>,
    },

    // Tensor operations
    // Index into a tensor: tensor[i] or tensor[i, j]
    Index {
        tensor: Box<Expr>,
        indices: Vec<Expr>,
    },
    // Reshape a tensor
    Reshape {
        expr: Box<Expr>,
        shape: Vec<Expr>,
    },
    // Reduce along a dimension: sum.dim(tensor), mean.dim(tensor)
    Reduce {
        op: ReduceOp,
        dim: Option<Dimension>,
        expr: Box<Expr>,
    },
    // Map a function along a dimension
    Map {
        dim: Option<Dimension>,
        func: Box<Expr>,
        expr: Box<Expr>,
    },
    // Dot product / matrix multiply
    Dot {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    // Matrix multiplication with optional dimension annotation
    // a @.batch b means batched matrix multiply
    // a @ b means standard matrix multiply
    MatMul {
        dim: Option<Dimension>,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    // Activation functions
    Relu(Box<Expr>),
    Softmax {
        dim: Option<Dimension>,
        expr: Box<Expr>,
    },
    LayerNorm {
        dim: Option<Dimension>,
        expr: Box<Expr>,
    },
    // Transpose dimensions
    Transpose {
        expr: Box<Expr>,
        dim1: Dimension,
        dim2: Dimension,
    },
    
    // Arithmetic and logical
    BinOp {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    UnOp {
        op: UnOp,
        expr: Box<Expr>,
    },
    
    // Control flow
    If {
        cond: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
    },
    
    // Functions
    Lambda {
        params: Vec<Ident>,
        body: Box<Expr>,
    },
    App {
        func: Box<Expr>,
        args: Vec<Expr>,
    },
    
    // Local bindings
    Let {
        bindings: Vec<(Ident, Expr)>,
        body: Box<Expr>,
    },
    Where {
        expr: Box<Expr>,
        bindings: Vec<(Ident, Expr)>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnOp {
    Neg,
    Not,
}

/// Reduction operations for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
}

impl fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ReduceOp::Sum => write!(f, "sum"),
            ReduceOp::Mean => write!(f, "mean"),
            ReduceOp::Max => write!(f, "max"),
            ReduceOp::Min => write!(f, "min"),
            ReduceOp::Prod => write!(f, "prod"),
        }
    }
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Eq => write!(f, "=="),
            BinOp::Neq => write!(f, "!="),
            BinOp::Lt => write!(f, "<"),
            BinOp::Lte => write!(f, "<="),
            BinOp::Gt => write!(f, ">"),
            BinOp::Gte => write!(f, ">="),
            BinOp::And => write!(f, "&&"),
            BinOp::Or => write!(f, "||"),
        }
    }
}

impl fmt::Display for UnOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            UnOp::Neg => write!(f, "-"),
            UnOp::Not => write!(f, "!"),
        }
    }
}

// Helper constructors
impl Expr {
    pub fn var(name: impl Into<String>) -> Self {
        Expr::Var(name.into())
    }

    pub fn int(n: i64) -> Self {
        Expr::Int(n)
    }

    pub fn tensor(elements: Vec<Expr>) -> Self {
        Expr::Tensor(elements)
    }

    pub fn fby(init: Expr, next: Expr) -> Self {
        Expr::Fby {
            dim: None,
            init: Box::new(init),
            next: Box::new(next),
        }
    }

    pub fn fby_dim(dim: Dimension, init: Expr, next: Expr) -> Self {
        Expr::Fby {
            dim: Some(dim),
            init: Box::new(init),
            next: Box::new(next),
        }
    }

    pub fn next(expr: Expr) -> Self {
        Expr::Next {
            dim: None,
            expr: Box::new(expr),
        }
    }

    pub fn next_dim(dim: Dimension, expr: Expr) -> Self {
        Expr::Next {
            dim: Some(dim),
            expr: Box::new(expr),
        }
    }

    pub fn prev(expr: Expr) -> Self {
        Expr::Prev {
            dim: None,
            expr: Box::new(expr),
        }
    }

    pub fn prev_dim(dim: Dimension, expr: Expr) -> Self {
        Expr::Prev {
            dim: Some(dim),
            expr: Box::new(expr),
        }
    }

    pub fn binop(op: BinOp, left: Expr, right: Expr) -> Self {
        Expr::BinOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn app(func: Expr, args: Vec<Expr>) -> Self {
        Expr::App {
            func: Box::new(func),
            args,
        }
    }

    pub fn dot(left: Expr, right: Expr) -> Self {
        Expr::Dot {
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn matmul(left: Expr, right: Expr) -> Self {
        Expr::MatMul {
            dim: None,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn matmul_dim(dim: Dimension, left: Expr, right: Expr) -> Self {
        Expr::MatMul {
            dim: Some(dim),
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn relu(expr: Expr) -> Self {
        Expr::Relu(Box::new(expr))
    }

    pub fn softmax(expr: Expr) -> Self {
        Expr::Softmax {
            dim: None,
            expr: Box::new(expr),
        }
    }

    pub fn softmax_dim(dim: Dimension, expr: Expr) -> Self {
        Expr::Softmax {
            dim: Some(dim),
            expr: Box::new(expr),
        }
    }

    pub fn layer_norm(expr: Expr) -> Self {
        Expr::LayerNorm {
            dim: None,
            expr: Box::new(expr),
        }
    }

    pub fn reduce(op: ReduceOp, dim: Option<Dimension>, expr: Expr) -> Self {
        Expr::Reduce {
            op,
            dim,
            expr: Box::new(expr),
        }
    }

    pub fn index(tensor: Expr, indices: Vec<Expr>) -> Self {
        Expr::Index {
            tensor: Box::new(tensor),
            indices,
        }
    }
}
