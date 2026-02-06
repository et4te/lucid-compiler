use crate::dataflow::*;
use crate::demand_analysis::DemandAnalysis;
use crate::cse::CSE;
use crate::loop_fusion::{LoopFusion, BufferMinimization};
use std::collections::{HashMap, HashSet};

pub struct Optimizer {
    pub demand_analysis: Option<DemandAnalysis>,
    pub loop_fusion: Option<LoopFusion>,
    pub buffer_minimization: Option<BufferMinimization>,
    pub cse_mapping: HashMap<NodeId, NodeId>,
}

impl Optimizer {
    pub fn new() -> Self {
        Optimizer {
            demand_analysis: None,
            loop_fusion: None,
            buffer_minimization: None,
            cse_mapping: HashMap::new(),
        }
    }
    
    /// Run all optimization passes on a dataflow graph
    pub fn optimize(&mut self, graph: &mut DataflowGraph) -> OptimizationResult {
        let mut result = OptimizationResult::default();
        
        // Phase 1: Common Subexpression Elimination
        println!("Running CSE...");
        let cse = CSE::new();
        let nodes_before = graph.nodes.len();
        self.cse_mapping = cse.optimize(graph);
        result.cse_eliminations = self.cse_mapping.len();
        println!("  Eliminated {} duplicate expressions", result.cse_eliminations);
        
        // Phase 2: Demand Analysis
        println!("Running demand analysis...");
        let demand = DemandAnalysis::analyze(graph, graph.entry_point);
        let dead_nodes = demand.dead_nodes(graph);
        result.dead_code_eliminated = dead_nodes.len();
        
        // Remove dead nodes
        for node_id in &dead_nodes {
            graph.nodes.remove(node_id);
            graph.edges.remove(node_id);
        }
        println!("  Eliminated {} dead nodes", result.dead_code_eliminated);
        
        self.demand_analysis = Some(demand);
        
        // Phase 3: Loop Fusion
        println!("Running loop fusion...");
        let fusion = LoopFusion::new().analyze(graph);
        result.fusion_groups = fusion.get_schedule().len();
        println!("  Created {} fusion groups", result.fusion_groups);
        
        self.loop_fusion = Some(fusion);
        
        // Phase 4: Buffer Minimization
        println!("Running buffer minimization...");
        if let Some(ref fusion) = self.loop_fusion {
            let buffer_min = BufferMinimization::new().analyze(graph, fusion);
            
            let total_buffer: usize = buffer_min.get_all_sizes().values().sum();
            let avg_buffer = if !buffer_min.get_all_sizes().is_empty() {
                total_buffer as f64 / buffer_min.get_all_sizes().len() as f64
            } else {
                0.0
            };
            
            result.total_buffer_size = total_buffer;
            result.avg_buffer_size = avg_buffer;
            
            println!("  Total buffer size: {} values", total_buffer);
            println!("  Average buffer size: {:.2} values", avg_buffer);
            
            self.buffer_minimization = Some(buffer_min);
        }
        
        result.nodes_after = graph.nodes.len();
        
        println!("\nOptimization complete!");
        println!("  Nodes: {} -> {}", nodes_before, result.nodes_after);
        
        result
    }
    
    /// Get buffer size requirement for a node
    pub fn get_buffer_size(&self, node_id: NodeId) -> usize {
        if let Some(ref buffer_min) = self.buffer_minimization {
            buffer_min.get_buffer_size(node_id)
        } else if let Some(ref demand) = self.demand_analysis {
            demand.buffer_size(node_id)
        } else {
            1 // Default minimum
        }
    }
    
    /// Check if two nodes can be fused
    pub fn can_fuse(&self, node1: NodeId, node2: NodeId) -> bool {
        if let Some(ref fusion) = self.loop_fusion {
            fusion.can_fuse(node1, node2)
        } else {
            false
        }
    }
    
    /// Get all nodes that are demanded
    pub fn get_demanded_nodes(&self) -> Option<&HashSet<NodeId>> {
        self.demand_analysis.as_ref().map(|d| &d.demanded)
    }
    
    /// Print optimization statistics
    pub fn print_stats(&self, graph: &DataflowGraph) {
        println!("\n=== Optimization Statistics ===");
        
        if let Some(ref demand) = self.demand_analysis {
            println!("Demanded nodes: {}", demand.demanded.len());
            println!("Total nodes: {}", graph.nodes.len());
        }
        
        if let Some(ref fusion) = self.loop_fusion {
            println!("Fusion groups: {}", fusion.get_schedule().len());
            for (i, group) in fusion.get_schedule().iter().enumerate() {
                println!("  Group {}: {} nodes", i, group.len());
            }
        }
        
        if let Some(ref buffer_min) = self.buffer_minimization {
            println!("Buffer requirements:");
            for (node_id, size) in buffer_min.get_all_sizes() {
                println!("  Node {}: {} values", node_id, size);
            }
        }
        
        println!("CSE eliminations: {}", self.cse_mapping.len());
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Default)]
pub struct OptimizationResult {
    pub cse_eliminations: usize,
    pub dead_code_eliminated: usize,
    pub fusion_groups: usize,
    pub total_buffer_size: usize,
    pub avg_buffer_size: f64,
    pub nodes_after: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::BinOp;
    
    #[test]
    fn test_full_optimization() {
        let mut graph = DataflowGraph::new();
        
        // Create a simple graph with redundancy
        let a = graph.add_node(Node::Constant(Value::Int(5)));
        let b = graph.add_node(Node::Constant(Value::Int(3)));
        
        // Two identical additions (will be CSE'd)
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
        
        // A dead node (not reachable from entry)
        let dead = graph.add_node(Node::Constant(Value::Int(999)));
        
        graph.entry_point = add1;
        
        let mut optimizer = Optimizer::new();
        let result = optimizer.optimize(&mut graph);
        
        assert!(result.cse_eliminations > 0 || result.dead_code_eliminated > 0);
    }
}
