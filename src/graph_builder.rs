use crate::ast::*;
use crate::dataflow::*;
use std::collections::HashMap;

pub struct GraphBuilder {
    graph: DataflowGraph,
    env: HashMap<String, NodeId>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        GraphBuilder {
            graph: DataflowGraph::new(),
            env: HashMap::new(),
        }
    }

    pub fn build(mut self, program: &Program) -> DataflowGraph {
        // Process definitions
        for def in &program.definitions {
            match def {
                Definition::Function { name, params, body } => {
                    let body_node = self.build_expr(body);
                    let lambda_node = self.graph.add_node(Node::Lambda {
                        params: params.clone(),
                        body: body_node,
                    });
                    self.env.insert(name.clone(), lambda_node);
                }
                Definition::Equation { name, expr } => {
                    let node = self.build_expr(expr);
                    self.env.insert(name.clone(), node);
                }
            }
        }

        // Build main expression
        let entry = self.build_expr(&program.main_expr);
        self.graph.entry_point = entry;

        self.graph
    }

    fn build_expr(&mut self, expr: &Expr) -> NodeId {
        match expr {
            Expr::Int(n) => self.graph.add_node(Node::Constant(Value::Int(*n))),

            Expr::Float(f) => self.graph.add_node(Node::Constant(Value::Float(*f))),

            Expr::Bool(b) => self.graph.add_node(Node::Constant(Value::Bool(*b))),

            Expr::Var(name) => {
                if let Some(&node_id) = self.env.get(name) {
                    node_id
                } else {
                    self.graph.add_node(Node::Input(name.clone()))
                }
            }

            Expr::Tensor(elements) => {
                let element_nodes: Vec<NodeId> =
                    elements.iter().map(|e| self.build_expr(e)).collect();
                self.graph.add_node(Node::TensorLiteral(element_nodes))
            }

            Expr::Fby { dim, init, next } => {
                let init_node = self.build_expr(init);
                let next_node = self.build_expr(next);

                // Extract initial value if it's a constant
                let init_val = match self.graph.nodes.get(&init_node) {
                    Some(Node::Constant(v)) => v.clone(),
                    _ => Value::Int(0), // Default
                };

                self.graph.add_node(Node::Delay {
                    input: next_node,
                    amount: 1,
                    init: init_val,
                    dim: dim.as_ref().map(|d| d.0.clone()),
                })
            }

            Expr::Next { dim, expr } => {
                let input_node = self.build_expr(expr);
                // Next is implemented as looking ahead
                // For now, simplified to just return the input
                // A full implementation would need special handling
                let _ = dim; // Mark as used
                input_node
            }

            Expr::Prev { dim, expr } => {
                let input_node = self.build_expr(expr);
                self.graph.add_node(Node::Delay {
                    input: input_node,
                    amount: 1,
                    init: Value::Int(0),
                    dim: dim.as_ref().map(|d| d.0.clone()),
                })
            }

            Expr::First { dim, expr } => {
                // First extracts the initial value of a stream
                let _ = dim; // Mark as used
                self.build_expr(expr)
            }

            Expr::Index { tensor, indices } => {
                let tensor_node = self.build_expr(tensor);
                let index_nodes: Vec<NodeId> =
                    indices.iter().map(|i| self.build_expr(i)).collect();
                self.graph.add_node(Node::Index {
                    tensor: tensor_node,
                    indices: index_nodes,
                })
            }

            Expr::Reshape { expr, shape } => {
                let input_node = self.build_expr(expr);
                let shape_nodes: Vec<NodeId> =
                    shape.iter().map(|s| self.build_expr(s)).collect();
                self.graph.add_node(Node::Reshape {
                    input: input_node,
                    shape: shape_nodes,
                })
            }

            Expr::Reduce { op, dim, expr } => {
                let input_node = self.build_expr(expr);
                self.graph.add_node(Node::Reduce {
                    op: *op,
                    dim: dim.as_ref().map(|d| d.0.clone()),
                    input: input_node,
                })
            }

            Expr::Map { dim, func, expr } => {
                let func_node = self.build_expr(func);
                let input_node = self.build_expr(expr);
                // Map is implemented as Apply over a dimension
                let _ = dim; // Mark as used
                self.graph.add_node(Node::Apply {
                    func: func_node,
                    args: vec![input_node],
                })
            }

            Expr::Dot { left, right } => {
                let left_node = self.build_expr(left);
                let right_node = self.build_expr(right);
                self.graph.add_node(Node::Dot {
                    left: left_node,
                    right: right_node,
                })
            }

            Expr::MatMul { dim, left, right } => {
                let left_node = self.build_expr(left);
                let right_node = self.build_expr(right);
                self.graph.add_node(Node::MatMul {
                    dim: dim.as_ref().map(|d| d.0.clone()),
                    left: left_node,
                    right: right_node,
                })
            }

            Expr::Relu(expr) => {
                let input_node = self.build_expr(expr);
                self.graph.add_node(Node::Relu { input: input_node })
            }

            Expr::Softmax { dim, expr } => {
                let input_node = self.build_expr(expr);
                self.graph.add_node(Node::Softmax {
                    dim: dim.as_ref().map(|d| d.0.clone()),
                    input: input_node,
                })
            }

            Expr::LayerNorm { dim, expr } => {
                let input_node = self.build_expr(expr);
                self.graph.add_node(Node::LayerNorm {
                    dim: dim.as_ref().map(|d| d.0.clone()),
                    input: input_node,
                })
            }

            Expr::Transpose { expr, dim1, dim2 } => {
                let input_node = self.build_expr(expr);
                self.graph.add_node(Node::Transpose {
                    input: input_node,
                    dim1: dim1.0.clone(),
                    dim2: dim2.0.clone(),
                })
            }

            Expr::BinOp { op, left, right } => {
                let left_node = self.build_expr(left);
                let right_node = self.build_expr(right);
                self.graph.add_node(Node::BinOp {
                    op: *op,
                    left: left_node,
                    right: right_node,
                })
            }

            Expr::UnOp { op, expr } => {
                let operand = self.build_expr(expr);
                self.graph.add_node(Node::UnOp { op: *op, operand })
            }

            Expr::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond_node = self.build_expr(cond);
                let then_node = self.build_expr(then_branch);
                let else_node = self.build_expr(else_branch);
                self.graph.add_node(Node::Select {
                    cond: cond_node,
                    then_val: then_node,
                    else_val: else_node,
                })
            }

            Expr::Lambda { params, body } => {
                let saved_env = self.env.clone();

                // Clear params from env to avoid capture issues
                for param in params {
                    self.env.remove(param);
                }

                let body_node = self.build_expr(body);

                self.env = saved_env;

                self.graph.add_node(Node::Lambda {
                    params: params.clone(),
                    body: body_node,
                })
            }

            Expr::App { func, args } => {
                let func_node = self.build_expr(func);
                let arg_nodes: Vec<NodeId> = args.iter().map(|arg| self.build_expr(arg)).collect();

                self.graph.add_node(Node::Apply {
                    func: func_node,
                    args: arg_nodes,
                })
            }

            Expr::Let { bindings, body } => {
                let saved_env = self.env.clone();

                for (name, expr) in bindings {
                    let node = self.build_expr(expr);
                    self.env.insert(name.clone(), node);
                }

                let result = self.build_expr(body);

                self.env = saved_env;
                result
            }

            Expr::Where { expr, bindings } => {
                let saved_env = self.env.clone();

                // Process bindings (may be mutually recursive)
                let mut binding_nodes = HashMap::new();
                for (name, _) in bindings {
                    // Create placeholder nodes for recursive references
                    let placeholder = self.graph.add_node(Node::Input(name.clone()));
                    self.env.insert(name.clone(), placeholder);
                    binding_nodes.insert(name.clone(), placeholder);
                }

                // Now build actual expressions
                for (name, expr) in bindings {
                    let node = self.build_expr(expr);
                    // Update the binding
                    self.env.insert(name.clone(), node);
                    if let Some(&placeholder) = binding_nodes.get(name) {
                        // Replace references to placeholder with actual node
                        if let Some(actual_node) = self.graph.nodes.get(&node).cloned() {
                            self.graph.nodes.insert(placeholder, actual_node);
                        }
                    }
                }

                let result = self.build_expr(expr);

                self.env = saved_env;
                result
            }
        }
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub fn ast_to_dataflow(program: &Program) -> DataflowGraph {
    let builder = GraphBuilder::new();
    builder.build(program)
}
