use crate::dataflow::*;
use std::collections::{HashMap, HashSet};

/// Demand analysis determines which nodes are actually needed for computation
/// This enables dead code elimination and lazy evaluation
pub struct DemandAnalysis {
    pub demanded: HashSet<NodeId>,
    pub demand_depth: HashMap<NodeId, usize>, // How many future values are needed
}

impl DemandAnalysis {
    pub fn analyze(graph: &DataflowGraph, entry: NodeId) -> Self {
        let mut analysis = DemandAnalysis {
            demanded: HashSet::new(),
            demand_depth: HashMap::new(),
        };
        
        analysis.mark_demanded(graph, entry, 0);
        analysis
    }
    
    fn mark_demanded(&mut self, graph: &DataflowGraph, node_id: NodeId, depth: usize) {
        if self.demanded.contains(&node_id) {
            // Update depth if we need more values
            if let Some(current_depth) = self.demand_depth.get(&node_id) {
                if depth > *current_depth {
                    self.demand_depth.insert(node_id, depth);
                    // Re-analyze dependencies with new depth
                    self.analyze_node(graph, node_id, depth);
                }
            }
            return;
        }
        
        self.demanded.insert(node_id);
        self.demand_depth.insert(node_id, depth);
        self.analyze_node(graph, node_id, depth);
    }
    
    fn analyze_node(&mut self, graph: &DataflowGraph, node_id: NodeId, depth: usize) {
        match graph.nodes.get(&node_id) {
            Some(Node::Delay { input, amount, .. }) => {
                // Delay consumes one time step, so we need depth + amount values from input
                self.mark_demanded(graph, *input, depth + amount);
            }
            
            Some(Node::BinOp { left, right, .. }) => {
                self.mark_demanded(graph, *left, depth);
                self.mark_demanded(graph, *right, depth);
            }
            
            Some(Node::UnOp { operand, .. }) => {
                self.mark_demanded(graph, *operand, depth);
            }
            
            Some(Node::Select { cond, then_val, else_val }) => {
                self.mark_demanded(graph, *cond, depth);
                // Both branches might be needed
                self.mark_demanded(graph, *then_val, depth);
                self.mark_demanded(graph, *else_val, depth);
            }
            
            Some(Node::Apply { func, args }) => {
                self.mark_demanded(graph, *func, depth);
                for &arg in args {
                    self.mark_demanded(graph, arg, depth);
                }
            }
            
            Some(Node::Lambda { body, .. }) => {
                // Body is only demanded when function is applied
                // For now, conservatively mark as demanded
                self.mark_demanded(graph, *body, depth);
            }
            
            Some(Node::Memo { expr }) => {
                self.mark_demanded(graph, *expr, depth);
            }
            
            _ => {}
        }
    }
    
    /// Returns nodes that can be eliminated (not demanded)
    pub fn dead_nodes(&self, graph: &DataflowGraph) -> HashSet<NodeId> {
        let all_nodes: HashSet<NodeId> = graph.nodes.keys().copied().collect();
        all_nodes.difference(&self.demanded).copied().collect()
    }
    
    /// Calculate how much buffering each node needs
    pub fn buffer_size(&self, node_id: NodeId) -> usize {
        self.demand_depth.get(&node_id).copied().unwrap_or(0) + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_demand_analysis() {
        let mut graph = DataflowGraph::new();
        
        // Build a simple graph: delay(const(1))
        let const_node = graph.add_node(Node::Constant(Value::Int(1)));
        let delay_node = graph.add_node(Node::Delay {
            input: const_node,
            amount: 1,
            init: Value::Int(0),
            dim: None,
        });
        
        let analysis = DemandAnalysis::analyze(&graph, delay_node);
        
        assert!(analysis.demanded.contains(&delay_node));
        assert!(analysis.demanded.contains(&const_node));
        assert_eq!(analysis.buffer_size(const_node), 2); // Needs 1 + delay amount
    }
}
