use crate::dataflow::*;
use std::collections::HashMap;

/// Common Subexpression Elimination identifies and eliminates duplicate computations
pub struct CSE {
    // Maps expression signatures to their node IDs
    expr_cache: HashMap<ExprSignature, NodeId>,
    // Maps old node IDs to new (deduplicated) node IDs
    node_mapping: HashMap<NodeId, NodeId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ExprSignature {
    Constant(String), // String representation of value
    BinOp {
        op: crate::ast::BinOp,
        left: NodeId,
        right: NodeId,
    },
    UnOp {
        op: crate::ast::UnOp,
        operand: NodeId,
    },
    Delay {
        input: NodeId,
        amount: usize,
    },
    Select {
        cond: NodeId,
        then_val: NodeId,
        else_val: NodeId,
    },
}

impl CSE {
    pub fn new() -> Self {
        CSE {
            expr_cache: HashMap::new(),
            node_mapping: HashMap::new(),
        }
    }
    
    pub fn optimize(mut self, graph: &mut DataflowGraph) -> HashMap<NodeId, NodeId> {
        let nodes: Vec<NodeId> = graph.nodes.keys().copied().collect();
        
        // Process in topological order to ensure dependencies are processed first
        let reachable = graph.reachable(graph.entry_point);
        let ordered = graph.topological_sort(&reachable);
        
        for node_id in ordered {
            self.process_node(graph, node_id);
        }
        
        // Apply mappings to update all node references
        self.apply_mappings(graph);
        
        self.node_mapping
    }
    
    fn process_node(&mut self, graph: &DataflowGraph, node_id: NodeId) {
        let signature = match graph.nodes.get(&node_id) {
            Some(Node::Constant(val)) => {
                Some(ExprSignature::Constant(format!("{:?}", val)))
            }
            Some(Node::BinOp { op, left, right }) => {
                let left = self.resolve(*left);
                let right = self.resolve(*right);
                Some(ExprSignature::BinOp { op: *op, left, right })
            }
            Some(Node::UnOp { op, operand }) => {
                let operand = self.resolve(*operand);
                Some(ExprSignature::UnOp { op: *op, operand })
            }
            Some(Node::Delay { input, amount, .. }) => {
                let input = self.resolve(*input);
                Some(ExprSignature::Delay {
                    input,
                    amount: *amount,
                })
            }
            Some(Node::Select { cond, then_val, else_val }) => {
                let cond = self.resolve(*cond);
                let then_val = self.resolve(*then_val);
                let else_val = self.resolve(*else_val);
                Some(ExprSignature::Select {
                    cond,
                    then_val,
                    else_val,
                })
            }
            _ => None, // Don't cache inputs, lambdas, etc.
        };
        
        if let Some(sig) = signature {
            if let Some(&existing) = self.expr_cache.get(&sig) {
                // Found a duplicate! Map this node to the existing one
                self.node_mapping.insert(node_id, existing);
            } else {
                // First occurrence, cache it
                self.expr_cache.insert(sig, node_id);
            }
        }
    }
    
    fn resolve(&self, node_id: NodeId) -> NodeId {
        self.node_mapping.get(&node_id).copied().unwrap_or(node_id)
    }
    
    fn apply_mappings(&self, graph: &mut DataflowGraph) {
        let nodes: Vec<(NodeId, Node)> = graph.nodes.iter()
            .map(|(&id, node)| (id, node.clone()))
            .collect();
        
        for (node_id, node) in nodes {
            let updated = match node {
                Node::BinOp { op, left, right } => Some(Node::BinOp {
                    op,
                    left: self.resolve(left),
                    right: self.resolve(right),
                }),
                Node::UnOp { op, operand } => Some(Node::UnOp {
                    op,
                    operand: self.resolve(operand),
                }),
                Node::Delay { input, amount, init, dim } => Some(Node::Delay {
                    input: self.resolve(input),
                    amount,
                    dim: dim.clone(),
                    init,
                }),
                Node::Select { cond, then_val, else_val } => Some(Node::Select {
                    cond: self.resolve(cond),
                    then_val: self.resolve(then_val),
                    else_val: self.resolve(else_val),
                }),
                Node::Apply { func, args } => Some(Node::Apply {
                    func: self.resolve(func),
                    args: args.iter().map(|&id| self.resolve(id)).collect(),
                }),
                Node::Memo { expr } => Some(Node::Memo {
                    expr: self.resolve(expr),
                }),
                _ => None,
            };
            
            if let Some(updated_node) = updated {
                graph.nodes.insert(node_id, updated_node);
            }
        }
        
        // Update entry point
        graph.entry_point = self.resolve(graph.entry_point);
    }
}

impl Default for CSE {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::BinOp;
    
    #[test]
    fn test_cse() {
        let mut graph = DataflowGraph::new();

        let a = graph.add_node(Node::Constant(Value::Int(5)));
        let b = graph.add_node(Node::Constant(Value::Int(3)));

        // Two identical additions
        let add1 = graph.add_node(Node::BinOp {
            op: BinOp::Add,
            left: a,
            right: b,
        });
        let add2 = graph.add_node(Node::BinOp {
            op: BinOp::Add,
            left: a,
            right: b,
        });

        // Create a result that uses BOTH adds, making them both reachable
        let result = graph.add_node(Node::BinOp {
            op: BinOp::Mul,
            left: add1,
            right: add2,
        });

        graph.entry_point = result;

        let cse = CSE::new();
        let mapping = cse.optimize(&mut graph);

        // One of the additions should be mapped to the other (order is non-deterministic)
        let deduped = mapping.get(&add1).is_some() || mapping.get(&add2).is_some();
        assert!(deduped, "CSE should deduplicate identical additions");

        // After CSE, the result node should reference the same add node for both operands
        if let Some(Node::BinOp { left, right, .. }) = graph.nodes.get(&result) {
            assert_eq!(left, right, "After CSE, both operands should reference the same node");
        }
    }
}
