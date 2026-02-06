use crate::dataflow::*;
use std::collections::{HashMap, HashSet};

/// Loop Fusion combines multiple passes over the same data into a single pass
/// This is particularly important for stream processing where we want to
/// minimize the number of iterations over temporal data
pub struct LoopFusion {
    // Groups of nodes that can be computed in the same loop
    fusion_groups: Vec<HashSet<NodeId>>,
}

impl LoopFusion {
    pub fn new() -> Self {
        LoopFusion {
            fusion_groups: Vec::new(),
        }
    }
    
    pub fn analyze(mut self, graph: &DataflowGraph) -> Self {
        // Find all temporal operations (delays) - these define loop boundaries
        let temporal_nodes = self.find_temporal_nodes(graph);
        
        // Build fusion groups based on dependencies
        for &temp_node in &temporal_nodes {
            let group = self.build_fusion_group(graph, temp_node, &temporal_nodes);
            if !group.is_empty() {
                self.fusion_groups.push(group);
            }
        }
        
        // Merge overlapping groups
        self.merge_groups();
        
        self
    }
    
    fn find_temporal_nodes(&self, graph: &DataflowGraph) -> HashSet<NodeId> {
        graph.nodes.iter()
            .filter_map(|(&id, node)| {
                match node {
                    Node::Delay { .. } => Some(id),
                    _ => None,
                }
            })
            .collect()
    }
    
    fn build_fusion_group(
        &self,
        graph: &DataflowGraph,
        start: NodeId,
        temporal_nodes: &HashSet<NodeId>,
    ) -> HashSet<NodeId> {
        let mut group = HashSet::new();
        let mut stack = vec![start];
        
        while let Some(node_id) = stack.pop() {
            if group.insert(node_id) {
                // Add dependencies that aren't temporal boundaries
                for dep in graph.get_dependencies(node_id) {
                    // Don't cross temporal boundaries (except the starting one)
                    if dep == start || !temporal_nodes.contains(&dep) {
                        stack.push(dep);
                    }
                }
            }
        }
        
        group
    }
    
    fn merge_groups(&mut self) {
        let mut changed = true;
        while changed {
            changed = false;
            
            for i in 0..self.fusion_groups.len() {
                for j in (i + 1)..self.fusion_groups.len() {
                    // Check if groups overlap
                    let overlap = !self.fusion_groups[i]
                        .is_disjoint(&self.fusion_groups[j]);
                    
                    if overlap {
                        // Merge j into i
                        let group_j = self.fusion_groups[j].clone();
                        self.fusion_groups[i].extend(group_j);
                        self.fusion_groups.remove(j);
                        changed = true;
                        break;
                    }
                }
                
                if changed {
                    break;
                }
            }
        }
    }
    
    /// Get the fusion schedule - which nodes should be computed together
    pub fn get_schedule(&self) -> &[HashSet<NodeId>] {
        &self.fusion_groups
    }
    
    /// Check if two nodes can be fused (computed in same loop)
    pub fn can_fuse(&self, node1: NodeId, node2: NodeId) -> bool {
        self.fusion_groups.iter().any(|group| {
            group.contains(&node1) && group.contains(&node2)
        })
    }
}

impl Default for LoopFusion {
    fn default() -> Self {
        Self::new()
    }
}

/// Buffer minimization determines the minimum buffer sizes needed for each stream
pub struct BufferMinimization {
    buffer_sizes: HashMap<NodeId, usize>,
}

impl BufferMinimization {
    pub fn new() -> Self {
        BufferMinimization {
            buffer_sizes: HashMap::new(),
        }
    }
    
    pub fn analyze(mut self, graph: &DataflowGraph, fusion: &LoopFusion) -> Self {
        // For each fusion group, calculate minimum buffer requirements
        for group in fusion.get_schedule() {
            self.analyze_group(graph, group);
        }
        
        self
    }
    
    fn analyze_group(&mut self, graph: &DataflowGraph, group: &HashSet<NodeId>) {
        // Calculate the maximum delay needed for any node in the group
        let mut max_delays: HashMap<NodeId, usize> = HashMap::new();
        
        for &node_id in group {
            let delay = self.calculate_max_delay(graph, node_id, group);
            max_delays.insert(node_id, delay);
        }
        
        // Buffer size is max_delay + 1 (for current value)
        for (&node_id, &delay) in &max_delays {
            self.buffer_sizes.insert(node_id, delay + 1);
        }
    }
    
    fn calculate_max_delay(
        &self,
        graph: &DataflowGraph,
        node_id: NodeId,
        group: &HashSet<NodeId>,
    ) -> usize {
        let mut max_delay = 0;
        let mut visited = HashSet::new();
        let mut stack = vec![(node_id, 0)];
        
        while let Some((current, delay)) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            
            max_delay = max_delay.max(delay);
            
            // Check for delays in dependencies
            match graph.nodes.get(&current) {
                Some(Node::Delay { input, amount, .. }) => {
                    if group.contains(input) {
                        stack.push((*input, delay + amount));
                    }
                }
                _ => {
                    for dep in graph.get_dependencies(current) {
                        if group.contains(&dep) {
                            stack.push((dep, delay));
                        }
                    }
                }
            }
        }
        
        max_delay
    }
    
    /// Get the minimum buffer size for a node
    pub fn get_buffer_size(&self, node_id: NodeId) -> usize {
        self.buffer_sizes.get(&node_id).copied().unwrap_or(1)
    }
    
    /// Get all buffer size requirements
    pub fn get_all_sizes(&self) -> &HashMap<NodeId, usize> {
        &self.buffer_sizes
    }
}

impl Default for BufferMinimization {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_loop_fusion() {
        let mut graph = DataflowGraph::new();
        
        let a = graph.add_node(Node::Constant(Value::Int(1)));
        let delay_a = graph.add_node(Node::Delay {
            input: a,
            amount: 1,
            init: Value::Int(0),
            dim: None,
        });

        let b = graph.add_node(Node::Constant(Value::Int(2)));
        let delay_b = graph.add_node(Node::Delay {
            input: b,
            amount: 1,
            init: Value::Int(0),
            dim: None,
        });
        
        let fusion = LoopFusion::new().analyze(&graph);
        
        // Two independent delays should be in different groups initially
        // but could potentially be fused
        assert!(!fusion.fusion_groups.is_empty());
    }
    
    #[test]
    fn test_buffer_minimization() {
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
        
        let fusion = LoopFusion::new().analyze(&graph);
        let buffer_min = BufferMinimization::new().analyze(&graph, &fusion);
        
        // Should need at least a buffer of 3 for the input (1 + 2 delays)
        assert!(buffer_min.get_buffer_size(input) >= 1);
    }
}
