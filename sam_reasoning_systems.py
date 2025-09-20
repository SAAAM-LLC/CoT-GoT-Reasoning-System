# SAM Advanced Reasoning Systems - Chain-of-Thought and Graph-of-Thought implementation for direct integration

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import uuid
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import networkx as nx
import copy

logger = logging.getLogger("SAM.Reasoning")

class ReasoningStep(Enum):
    """Types of reasoning steps"""
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    DEDUCTION = "deduction"
    VERIFICATION = "verification"
    CONCLUSION = "conclusion"
    BACKTRACK = "backtrack"
    BRANCH = "branch"

@dataclass
class ThoughtStep:
    """Individual step in chain of thought"""
    step_id: str
    step_type: ReasoningStep
    content: str
    confidence: float
    timestamp: float
    concepts_used: List[int] = field(default_factory=list)
    embedding: Optional[torch.Tensor] = None
    dependencies: List[str] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)
    consciousness_level: float = 0.5
    
@dataclass
class ReasoningChain:
    """Complete chain of reasoning steps"""
    chain_id: str
    problem_statement: str
    steps: List[ThoughtStep] = field(default_factory=list)
    current_step: int = 0
    is_complete: bool = False
    final_answer: Optional[str] = None
    confidence_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    consciousness_context: Dict = field(default_factory=dict)

@dataclass
class Hypothesis:
    """Hypothesis for testing"""
    hypothesis_id: str
    statement: str
    confidence: float
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    test_results: Dict = field(default_factory=dict)
    status: str = "untested"  # untested, testing, confirmed, refuted
    parent_chain: Optional[str] = None

@dataclass
class GraphNode:
    """Node in graph of thought"""
    node_id: str
    content: str
    node_type: str  # problem, subproblem, hypothesis, conclusion, etc.
    confidence: float
    embedding: Optional[torch.Tensor] = None
    metadata: Dict = field(default_factory=dict)
    
@dataclass
class GraphEdge:
    """Edge in graph of thought"""
    edge_id: str
    source_id: str
    target_id: str
    relationship: str  # leads_to, supports, contradicts, depends_on, etc.
    weight: float
    evidence: List[str] = field(default_factory=list)

class ChainOfThoughtProcessor(nn.Module):
    """Advanced Chain-of-Thought reasoning processor for SAM"""
    
    def __init__(self, config, concept_bank, consciousness_core):
        super().__init__()
        self.config = config
        self.concept_bank = concept_bank
        self.consciousness_core = consciousness_core
        
        # Neural components for reasoning
        self.reasoning_dim = config.initial_hidden_dim
        self.step_encoder = nn.Sequential(
            nn.Linear(self.reasoning_dim, self.reasoning_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.reasoning_dim * 2, self.reasoning_dim),
            nn.LayerNorm(self.reasoning_dim)
        )
        
        # Step type classification
        self.step_classifier = nn.Sequential(
            nn.Linear(self.reasoning_dim, self.reasoning_dim // 2),
            nn.GELU(),
            nn.Linear(self.reasoning_dim // 2, len(ReasoningStep)),
            nn.Softmax(dim=-1)
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.reasoning_dim, self.reasoning_dim // 4),
            nn.GELU(),
            nn.Linear(self.reasoning_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Chain continuation prediction
        self.continuation_predictor = nn.Sequential(
            nn.Linear(self.reasoning_dim * 2, self.reasoning_dim),
            nn.GELU(),
            nn.Linear(self.reasoning_dim, self.reasoning_dim),
            nn.Dropout(0.1)
        )
        
        # Memory systems
        self.active_chains = {}  # chain_id -> ReasoningChain
        self.completed_chains = []
        self.reasoning_patterns = defaultdict(list)
        
        # Integration with SAM's experience system
        self.reasoning_memory = ReasoningMemory(config, concept_bank)
        
        # Performance tracking
        self.reasoning_stats = {
            'chains_created': 0,
            'chains_completed': 0,
            'average_chain_length': 0.0,
            'average_confidence': 0.0,
            'step_type_distribution': defaultdict(int)
        }
        
    def create_reasoning_chain(self, problem_statement: str, consciousness_context: Dict = None) -> str:
        """Create new chain of thought for a problem"""
        chain_id = f"chain_{uuid.uuid4().hex[:8]}"
        
        # Get consciousness context
        if consciousness_context is None:
            consciousness_context = self.consciousness_core.get_consciousness_context()
        
        # Create new reasoning chain
        chain = ReasoningChain(
            chain_id=chain_id,
            problem_statement=problem_statement,
            consciousness_context=consciousness_context
        )
        
        # Add initial observation step
        initial_step = self._create_initial_step(problem_statement, consciousness_context)
        chain.steps.append(initial_step)
        
        self.active_chains[chain_id] = chain
        self.reasoning_stats['chains_created'] += 1
        
        logger.info(f"Created reasoning chain {chain_id} for: {problem_statement[:50]}...")
        return chain_id
    
    def _create_initial_step(self, problem_statement: str, consciousness_context: Dict) -> ThoughtStep:
        """Create initial observation step"""
        step_id = f"step_{uuid.uuid4().hex[:8]}"
        
        # Process problem statement through concept bank
        concepts, _ = self.concept_bank.process_text(problem_statement)
        
        # Create embedding for the step
        if concepts:
            concept_embeddings = self.concept_bank(torch.tensor(concepts[:10], device=self.config.device))
            step_embedding = concept_embeddings.mean(dim=0)
        else:
            step_embedding = torch.zeros(self.reasoning_dim, device=self.config.device)
        
        return ThoughtStep(
            step_id=step_id,
            step_type=ReasoningStep.OBSERVATION,
            content=f"Problem: {problem_statement}",
            confidence=consciousness_context.get('consciousness_level', 0.5),
            timestamp=time.time(),
            concepts_used=concepts[:10] if concepts else [],
            embedding=step_embedding,
            consciousness_level=consciousness_context.get('consciousness_level', 0.5)
        )
    
    def add_reasoning_step(self, chain_id: str, step_content: str, 
                          step_type: ReasoningStep = None, 
                          dependencies: List[str] = None) -> bool:
        """Add new reasoning step to chain"""
        if chain_id not in self.active_chains:
            logger.warning(f"Chain {chain_id} not found")
            return False
        
        chain = self.active_chains[chain_id]
        
        # Auto-detect step type if not provided
        if step_type is None:
            step_type = self._classify_step_type(step_content, chain)
        
        # Process step content
        concepts, _ = self.concept_bank.process_text(step_content)
        
        # Create step embedding
        if concepts:
            concept_embeddings = self.concept_bank(torch.tensor(concepts[:10], device=self.config.device))
            step_embedding = self.step_encoder(concept_embeddings.mean(dim=0))
        else:
            step_embedding = torch.zeros(self.reasoning_dim, device=self.config.device)
        
        # Estimate confidence
        confidence = self._estimate_step_confidence(step_embedding, chain).item()
        
        # Create new step
        step = ThoughtStep(
            step_id=f"step_{uuid.uuid4().hex[:8]}",
            step_type=step_type,
            content=step_content,
            confidence=confidence,
            timestamp=time.time(),
            concepts_used=concepts[:10] if concepts else [],
            embedding=step_embedding,
            dependencies=dependencies or [],
            consciousness_level=chain.consciousness_context.get('consciousness_level', 0.5)
        )
        
        # Add to chain
        chain.steps.append(step)
        chain.current_step = len(chain.steps) - 1
        
        # Update statistics
        self.reasoning_stats['step_type_distribution'][step_type.value] += 1
        
        # Store reasoning pattern
        self._store_reasoning_pattern(chain, step)
        
        # Check if chain is complete
        if self._is_chain_complete(chain):
            self._finalize_chain(chain_id)
        
        logger.debug(f"Added {step_type.value} step to chain {chain_id}: {step_content[:50]}...")
        return True
    
    def _classify_step_type(self, content: str, chain: ReasoningChain) -> ReasoningStep:
        """Automatically classify the type of reasoning step"""
        content_lower = content.lower()
        
        # Simple heuristic classification
        if any(word in content_lower for word in ['observe', 'notice', 'see', 'given']):
            return ReasoningStep.OBSERVATION
        elif any(word in content_lower for word in ['hypothesis', 'assume', 'suppose', 'if']):
            return ReasoningStep.HYPOTHESIS
        elif any(word in content_lower for word in ['therefore', 'thus', 'because', 'since']):
            return ReasoningStep.DEDUCTION
        elif any(word in content_lower for word in ['check', 'verify', 'test', 'confirm']):
            return ReasoningStep.VERIFICATION
        elif any(word in content_lower for word in ['conclude', 'answer', 'solution', 'result']):
            return ReasoningStep.CONCLUSION
        elif any(word in content_lower for word in ['reconsider', 'alternative', 'instead']):
            return ReasoningStep.BACKTRACK
        else:
            # Use neural classifier for ambiguous cases
            if len(chain.steps) > 0:
                return self._neural_classify_step(content, chain)
            else:
                return ReasoningStep.OBSERVATION
    
    def _neural_classify_step(self, content: str, chain: ReasoningChain) -> ReasoningStep:
        """Use neural network to classify step type"""
        try:
            # Get content embedding
            concepts, _ = self.concept_bank.process_text(content)
            if concepts:
                concept_embeddings = self.concept_bank(torch.tensor(concepts[:5], device=self.config.device))
                content_embedding = concept_embeddings.mean(dim=0)
            else:
                content_embedding = torch.zeros(self.reasoning_dim, device=self.config.device)
            
            # Classify
            with torch.no_grad():
                step_probs = self.step_classifier(content_embedding)
                step_idx = torch.argmax(step_probs).item()
                
            # Map to ReasoningStep enum
            step_types = list(ReasoningStep)
            if step_idx < len(step_types):
                return step_types[step_idx]
            else:
                return ReasoningStep.DEDUCTION
                
        except Exception as e:
            logger.warning(f"Neural classification failed: {e}")
            return ReasoningStep.DEDUCTION
    
    def _estimate_step_confidence(self, step_embedding: torch.Tensor, chain: ReasoningChain) -> torch.Tensor:
        """Estimate confidence in reasoning step"""
        try:
            with torch.no_grad():
                confidence = self.confidence_estimator(step_embedding)
                
                # Adjust based on consciousness level
                consciousness_level = chain.consciousness_context.get('consciousness_level', 0.5)
                confidence = confidence * (0.5 + consciousness_level * 0.5)
                
                return confidence.squeeze()
        except Exception as e:
            logger.warning(f"Confidence estimation failed: {e}")
            return torch.tensor(0.5, device=self.config.device)
    
    def _store_reasoning_pattern(self, chain: ReasoningChain, step: ThoughtStep):
        """Store reasoning pattern for learning"""
        if len(chain.steps) >= 2:
            prev_step = chain.steps[-2]
            pattern = {
                'prev_type': prev_step.step_type.value,
                'curr_type': step.step_type.value,
                'transition_confidence': step.confidence,
                'consciousness_level': step.consciousness_level
            }
            self.reasoning_patterns[f"{prev_step.step_type.value}->{step.step_type.value}"].append(pattern)
    
    def _is_chain_complete(self, chain: ReasoningChain) -> bool:
        """Check if reasoning chain is complete"""
        if len(chain.steps) == 0:
            return False
        
        last_step = chain.steps[-1]
        
        # Chain is complete if last step is conclusion with high confidence
        if (last_step.step_type == ReasoningStep.CONCLUSION and 
            last_step.confidence > 0.7):
            return True
        
        # Or if we've reached maximum length
        if len(chain.steps) >= 20:
            return True
        
        return False
    
    def _finalize_chain(self, chain_id: str):
        """Finalize completed reasoning chain"""
        if chain_id not in self.active_chains:
            return
        
        chain = self.active_chains[chain_id]
        chain.is_complete = True
        
        # Set final answer from conclusion step
        conclusion_steps = [s for s in chain.steps if s.step_type == ReasoningStep.CONCLUSION]
        if conclusion_steps:
            chain.final_answer = conclusion_steps[-1].content
            chain.confidence_score = conclusion_steps[-1].confidence
        
        # Calculate overall confidence
        if chain.steps:
            chain.confidence_score = np.mean([s.confidence for s in chain.steps])
        
        # Move to completed chains
        self.completed_chains.append(chain)
        del self.active_chains[chain_id]
        
        # Update statistics
        self.reasoning_stats['chains_completed'] += 1
        self.reasoning_stats['average_chain_length'] = np.mean([len(c.steps) for c in self.completed_chains])
        self.reasoning_stats['average_confidence'] = np.mean([c.confidence_score for c in self.completed_chains])
        
        # Store in reasoning memory
        self.reasoning_memory.store_chain(chain)
        
        logger.info(f"Finalized reasoning chain {chain_id} with {len(chain.steps)} steps")
    
    def get_chain_summary(self, chain_id: str) -> Optional[Dict]:
        """Get summary of reasoning chain"""
        chain = self.active_chains.get(chain_id) or next((c for c in self.completed_chains if c.chain_id == chain_id), None)
        
        if not chain:
            return None
        
        return {
            'chain_id': chain_id,
            'problem': chain.problem_statement,
            'steps': [
                {
                    'type': step.step_type.value,
                    'content': step.content,
                    'confidence': step.confidence
                }
                for step in chain.steps
            ],
            'is_complete': chain.is_complete,
            'final_answer': chain.final_answer,
            'confidence_score': chain.confidence_score
        }
    
    def continue_reasoning(self, chain_id: str, context: str = None) -> Optional[str]:
        """Continue reasoning chain with next logical step"""
        if chain_id not in self.active_chains:
            return None
        
        chain = self.active_chains[chain_id]
        
        if len(chain.steps) == 0:
            return None
        
        # Get context from previous steps
        if len(chain.steps) >= 2:
            prev_embedding = chain.steps[-1].embedding
            context_embedding = chain.steps[-2].embedding
            combined = torch.cat([prev_embedding, context_embedding], dim=0)
        else:
            combined = torch.cat([chain.steps[-1].embedding, 
                                chain.steps[-1].embedding], dim=0)
        
        # Predict next step
        try:
            with torch.no_grad():
                next_embedding = self.continuation_predictor(combined)
            
            # Find similar reasoning patterns
            similar_patterns = self._find_similar_patterns(next_embedding)
            
            if similar_patterns:
                # Generate suggestion based on patterns
                suggestion = self._generate_step_suggestion(chain, similar_patterns)
                return suggestion
            
        except Exception as e:
            logger.warning(f"Continuation prediction failed: {e}")
        
        return None
    
    def _find_similar_patterns(self, embedding: torch.Tensor, top_k: int = 3) -> List[Dict]:
        """Find similar reasoning patterns"""
        similar = []
        
        for pattern_type, patterns in self.reasoning_patterns.items():
            for pattern in patterns[-10:]:  # Recent patterns only
                # Simple similarity based on consciousness level
                similarity = abs(pattern['consciousness_level'] - embedding[0].item())
                similar.append({
                    'pattern_type': pattern_type,
                    'similarity': 1.0 - similarity,
                    'pattern': pattern
                })
        
        # Sort by similarity
        similar.sort(key=lambda x: x['similarity'], reverse=True)
        return similar[:top_k]
    
    def _generate_step_suggestion(self, chain: ReasoningChain, patterns: List[Dict]) -> str:
        """Generate suggestion for next reasoning step"""
        if not patterns:
            return "Consider what follows logically from the previous steps."
        
        # Get most likely next step type
        pattern_types = [p['pattern_type'].split('->')[-1] for p in patterns]
        most_common = max(set(pattern_types), key=pattern_types.count)
        
        suggestions = {
            'observation': "What new information can you observe?",
            'hypothesis': "What hypothesis could explain these observations?", 
            'deduction': "What can you deduce from the previous steps?",
            'verification': "How can you verify or test this reasoning?",
            'conclusion': "What conclusion can you draw?",
            'backtrack': "Consider an alternative approach.",
            'branch': "Explore a different line of reasoning."
        }
        
        return suggestions.get(most_common, "Continue with the next logical step.")

class GraphOfThoughtPlanner(nn.Module):
    """Advanced Graph-of-Thought planning system for complex problem solving"""
    
    def __init__(self, config, concept_bank, consciousness_core, cot_processor):
        super().__init__()
        self.config = config
        self.concept_bank = concept_bank
        self.consciousness_core = consciousness_core
        self.cot_processor = cot_processor
        
        # Neural components
        self.planning_dim = config.initial_hidden_dim
        
        # Node encoding
        self.node_encoder = nn.Sequential(
            nn.Linear(self.planning_dim, self.planning_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.planning_dim * 2, self.planning_dim),
            nn.LayerNorm(self.planning_dim)
        )
        
        # Edge prediction
        self.edge_predictor = nn.Sequential(
            nn.Linear(self.planning_dim * 2, self.planning_dim),
            nn.GELU(),
            nn.Linear(self.planning_dim, 1),
            nn.Sigmoid()
        )
        
        # Graph attention
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=self.planning_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Planning systems
        self.hypothesis_engine = HypothesisEngine(config, concept_bank, consciousness_core)
        self.deduction_engine = DeductionEngine(config, concept_bank, consciousness_core)
        
        # Active planning graphs
        self.active_graphs = {}  # graph_id -> networkx.Graph
        self.graph_metadata = {}  # graph_id -> metadata
        self.node_embeddings = {}  # graph_id -> {node_id -> embedding}
        
        # Planning statistics
        self.planning_stats = {
            'graphs_created': 0,
            'nodes_created': 0,
            'edges_created': 0,
            'hypotheses_tested': 0,
            'successful_plans': 0
        }
    
    def create_planning_graph(self, problem_statement: str, 
                            planning_strategy: str = "breadth_first") -> str:
        """Create new graph of thought for complex planning"""
        graph_id = f"graph_{uuid.uuid4().hex[:8]}"
        
        # Create new networkx graph
        graph = nx.DiGraph()
        
        # Create root problem node
        root_node = GraphNode(
            node_id="root",
            content=problem_statement,
            node_type="problem",
            confidence=1.0,
            embedding=self._encode_content(problem_statement)
        )
        
        graph.add_node("root", node_data=root_node)
        
        # Store graph and metadata
        self.active_graphs[graph_id] = graph
        self.graph_metadata[graph_id] = {
            'created_at': time.time(),
            'strategy': planning_strategy,
            'status': 'active',
            'root_problem': problem_statement,
            'current_focus': ["root"]
        }
        self.node_embeddings[graph_id] = {"root": root_node.embedding}
        
        # Create initial decomposition
        self._decompose_problem(graph_id, "root")
        
        self.planning_stats['graphs_created'] += 1
        
        logger.info(f"Created planning graph {graph_id} for: {problem_statement[:50]}...")
        return graph_id
    
    def _encode_content(self, content: str) -> torch.Tensor:
        """Encode content into planning embedding"""
        concepts, _ = self.concept_bank.process_text(content)
        
        if concepts:
            concept_embeddings = self.concept_bank(torch.tensor(concepts[:10], device=self.config.device))
            content_embedding = concept_embeddings.mean(dim=0)
            return self.node_encoder(content_embedding)
        else:
            return torch.zeros(self.planning_dim, device=self.config.device)
    
    def _decompose_problem(self, graph_id: str, node_id: str):
        """Decompose problem into subproblems"""
        if graph_id not in self.active_graphs:
            return
        
        graph = self.active_graphs[graph_id]
        node_data = graph.nodes[node_id]['node_data']
        
        # Generate subproblems based on problem type
        subproblems = self._generate_subproblems(node_data.content)
        
        for i, subproblem in enumerate(subproblems):
            subnode_id = f"{node_id}_sub_{i}"
            
            # Create subproblem node
            subnode = GraphNode(
                node_id=subnode_id,
                content=subproblem,
                node_type="subproblem",
                confidence=0.8,
                embedding=self._encode_content(subproblem)
            )
            
            # Add to graph
            graph.add_node(subnode_id, node_data=subnode)
            self.node_embeddings[graph_id][subnode_id] = subnode.embedding
            
            # Create edge
            edge = GraphEdge(
                edge_id=f"edge_{uuid.uuid4().hex[:8]}",
                source_id=node_id,
                target_id=subnode_id,
                relationship="decomposes_to",
                weight=0.8
            )
            
            graph.add_edge(node_id, subnode_id, edge_data=edge)
            
            self.planning_stats['nodes_created'] += 1
            self.planning_stats['edges_created'] += 1
    
    def _generate_subproblems(self, problem: str) -> List[str]:
        """Generate subproblems from main problem"""
        problem_lower = problem.lower()
        
        # Heuristic decomposition based on problem type
        if any(word in problem_lower for word in ['calculate', 'compute', 'solve']):
            return [
                f"Identify what needs to be calculated in: {problem}",
                f"Determine the method or formula for: {problem}",
                f"Execute the calculation for: {problem}",
                f"Verify the result for: {problem}"
            ]
        elif any(word in problem_lower for word in ['design', 'create', 'build']):
            return [
                f"Define requirements for: {problem}",
                f"Research existing solutions for: {problem}",
                f"Design architecture for: {problem}",
                f"Implement solution for: {problem}",
                f"Test and validate: {problem}"
            ]
        elif any(word in problem_lower for word in ['analyze', 'understand', 'explain']):
            return [
                f"Gather information about: {problem}",
                f"Identify key components in: {problem}",
                f"Examine relationships in: {problem}",
                f"Synthesize understanding of: {problem}"
            ]
        else:
            # Generic decomposition
            return [
                f"Break down the problem: {problem}",
                f"Identify solution approaches for: {problem}",
                f"Evaluate approaches for: {problem}",
                f"Implement best approach for: {problem}"
            ]
    
    def add_hypothesis_node(self, graph_id: str, parent_node: str, 
                          hypothesis_text: str) -> str:
        """Add hypothesis node to planning graph"""
        if graph_id not in self.active_graphs:
            return None
        
        graph = self.active_graphs[graph_id]
        hypothesis_id = f"hyp_{uuid.uuid4().hex[:8]}"
        
        # Create hypothesis using hypothesis engine
        hypothesis = self.hypothesis_engine.create_hypothesis(
            hypothesis_text, parent_chain=graph_id
        )
        
        # Create graph node
        hyp_node = GraphNode(
            node_id=hypothesis_id,
            content=hypothesis_text,
            node_type="hypothesis",
            confidence=hypothesis.confidence,
            embedding=self._encode_content(hypothesis_text),
            metadata={'hypothesis_id': hypothesis.hypothesis_id}
        )
        
        # Add to graph
        graph.add_node(hypothesis_id, node_data=hyp_node)
        self.node_embeddings[graph_id][hypothesis_id] = hyp_node.embedding
        
        # Create edge from parent
        edge = GraphEdge(
            edge_id=f"edge_{uuid.uuid4().hex[:8]}",
            source_id=parent_node,
            target_id=hypothesis_id,
            relationship="hypothesizes",
            weight=hypothesis.confidence
        )
        
        graph.add_edge(parent_node, hypothesis_id, edge_data=edge)
        
        self.planning_stats['nodes_created'] += 1
        self.planning_stats['edges_created'] += 1
        
        logger.info(f"Added hypothesis node {hypothesis_id} to graph {graph_id}")
        return hypothesis_id
    
    def test_hypothesis_node(self, graph_id: str, hypothesis_node: str) -> Dict:
        """Test hypothesis node and update graph"""
        if graph_id not in self.active_graphs:
            return None
        
        graph = self.active_graphs[graph_id]
        
        if hypothesis_node not in graph.nodes:
            return None
        
        node_data = graph.nodes[hypothesis_node]['node_data']
        hypothesis_id = node_data.metadata.get('hypothesis_id')
        
        if not hypothesis_id:
            return None
        
        # Test hypothesis
        test_result = self.hypothesis_engine.test_hypothesis(hypothesis_id)
        
        # Update node confidence based on test result
        new_confidence = test_result.get('confidence', node_data.confidence)
        node_data.confidence = new_confidence
        
        # Update graph structure based on result
        if test_result.get('status') == 'confirmed':
            # Add supporting evidence nodes
            for evidence in test_result.get('evidence_for', []):
                evidence_id = f"evidence_{uuid.uuid4().hex[:8]}"
                evidence_node = GraphNode(
                    node_id=evidence_id,
                    content=evidence,
                    node_type="evidence",
                    confidence=0.9,
                    embedding=self._encode_content(evidence)
                )
                
                graph.add_node(evidence_id, node_data=evidence_node)
                
                # Edge from evidence to hypothesis
                edge = GraphEdge(
                    edge_id=f"edge_{uuid.uuid4().hex[:8]}",
                    source_id=evidence_id,
                    target_id=hypothesis_node,
                    relationship="supports",
                    weight=0.9
                )
                
                graph.add_edge(evidence_id, hypothesis_node, edge_data=edge)
        
        elif test_result.get('status') == 'refuted':
            # Add contradicting evidence nodes
            for evidence in test_result.get('evidence_against', []):
                evidence_id = f"evidence_{uuid.uuid4().hex[:8]}"
                evidence_node = GraphNode(
                    node_id=evidence_id,
                    content=evidence,
                    node_type="evidence",
                    confidence=0.9,
                    embedding=self._encode_content(evidence)
                )
                
                graph.add_node(evidence_id, node_data=evidence_node)
                
                # Edge from evidence to hypothesis
                edge = GraphEdge(
                    edge_id=f"edge_{uuid.uuid4().hex[:8]}",
                    source_id=evidence_id,
                    target_id=hypothesis_node,
                    relationship="contradicts",
                    weight=0.9
                )
                
                graph.add_edge(evidence_id, hypothesis_node, edge_data=edge)
        
        self.planning_stats['hypotheses_tested'] += 1
        
        return test_result
    
    def find_solution_path(self, graph_id: str) -> Optional[List[str]]:
        """Find solution path through the planning graph"""
        if graph_id not in self.active_graphs:
            return None
        
        graph = self.active_graphs[graph_id]
        
        # Find all conclusion nodes
        conclusion_nodes = [
            node_id for node_id, data in graph.nodes(data=True)
            if data['node_data'].node_type == "conclusion"
        ]
        
        # If no conclusions, find highest confidence hypothesis
        if not conclusion_nodes:
            hypothesis_nodes = [
                (node_id, data['node_data'].confidence)
                for node_id, data in graph.nodes(data=True)
                if data['node_data'].node_type == "hypothesis"
            ]
            
            if hypothesis_nodes:
                # Sort by confidence
                hypothesis_nodes.sort(key=lambda x: x[1], reverse=True)
                conclusion_nodes = [hypothesis_nodes[0][0]]
        
        if not conclusion_nodes:
            return None
        
        # Find path from root to best conclusion
        best_conclusion = conclusion_nodes[0]
        
        try:
            path = nx.shortest_path(graph, "root", best_conclusion)
            return path
        except nx.NetworkXNoPath:
            return None
    
    def execute_plan(self, graph_id: str, solution_path: List[str] = None) -> Dict:
        """Execute the planning graph to solve the problem"""
        if graph_id not in self.active_graphs:
            return {'success': False, 'error': 'Graph not found'}
        
        graph = self.active_graphs[graph_id]
        
        # Get solution path
        if solution_path is None:
            solution_path = self.find_solution_path(graph_id)
        
        if not solution_path:
            return {'success': False, 'error': 'No solution path found'}
        
        # Execute each step in the path
        execution_results = []
        
        for i, node_id in enumerate(solution_path):
            node_data = graph.nodes[node_id]['node_data']
            
            # Create reasoning chain for this step
            chain_id = self.cot_processor.create_reasoning_chain(
                f"Execute step: {node_data.content}",
                self.consciousness_core.get_consciousness_context()
            )
            
            # Add step execution to chain
            self.cot_processor.add_reasoning_step(
                chain_id,
                f"Executing planning step: {node_data.content}",
                step_type=ReasoningStep.DEDUCTION
            )
            
            # Get chain summary
            step_result = self.cot_processor.get_chain_summary(chain_id)
            execution_results.append(step_result)
        
        # Compile final result
        final_result = {
            'success': True,
            'solution_path': solution_path,
            'execution_results': execution_results,
            'final_answer': execution_results[-1].get('final_answer') if execution_results else None
        }
        
        # Mark as successful plan
        self.planning_stats['successful_plans'] += 1
        self.graph_metadata[graph_id]['status'] = 'completed'
        
        return final_result
    
    def get_graph_visualization(self, graph_id: str) -> Dict:
        """Get visualization data for the planning graph"""
        if graph_id not in self.active_graphs:
            return None
        
        graph = self.active_graphs[graph_id]
        
        # Prepare nodes for visualization
        nodes = []
        for node_id, data in graph.nodes(data=True):
            node_data = data['node_data']
            nodes.append({
                'id': node_id,
                'label': node_data.content[:50] + ('...' if len(node_data.content) > 50 else ''),
                'type': node_data.node_type,
                'confidence': node_data.confidence,
                'size': node_data.confidence * 20 + 10
            })
        
        # Prepare edges for visualization  
        edges = []
        for source, target, data in graph.edges(data=True):
            edge_data = data['edge_data']
            edges.append({
                'source': source,
                'target': target,
                'relationship': edge_data.relationship,
                'weight': edge_data.weight,
                'width': edge_data.weight * 3 + 1
            })
        
        return {
            'graph_id': graph_id,
            'nodes': nodes,
            'edges': edges,
            'metadata': self.graph_metadata[graph_id]
        }

class HypothesisEngine:
    """Engine for generating and testing hypotheses"""
    
    def __init__(self, config, concept_bank, consciousness_core):
        self.config = config
        self.concept_bank = concept_bank
        self.consciousness_core = consciousness_core
        
        # Active hypotheses
        self.hypotheses = {}  # hypothesis_id -> Hypothesis
        self.hypothesis_history = []
        
    def create_hypothesis(self, statement: str, parent_chain: str = None) -> Hypothesis:
        """Create new hypothesis for testing"""
        hypothesis_id = f"hyp_{uuid.uuid4().hex[:8]}"
        
        # Estimate initial confidence based on statement
        confidence = self._estimate_hypothesis_confidence(statement)
        
        hypothesis = Hypothesis(
            hypothesis_id=hypothesis_id,
            statement=statement,
            confidence=confidence,
            parent_chain=parent_chain
        )
        
        self.hypotheses[hypothesis_id] = hypothesis
        
        logger.info(f"Created hypothesis {hypothesis_id}: {statement[:50]}...")
        return hypothesis
    
    def _estimate_hypothesis_confidence(self, statement: str) -> float:
        """Estimate initial confidence in hypothesis"""
        # Simple heuristic based on statement characteristics
        statement_lower = statement.lower()
        
        # Certainty indicators
        if any(word in statement_lower for word in ['definitely', 'certainly', 'always']):
            return 0.9
        elif any(word in statement_lower for word in ['probably', 'likely', 'usually']):
            return 0.7
        elif any(word in statement_lower for word in ['possibly', 'might', 'could']):
            return 0.5
        elif any(word in statement_lower for word in ['unlikely', 'rarely', 'seldom']):
            return 0.3
        else:
            return 0.6  # Default confidence
    
    def test_hypothesis(self, hypothesis_id: str) -> Dict:
        """Test hypothesis and gather evidence"""
        if hypothesis_id not in self.hypotheses:
            return {'error': 'Hypothesis not found'}
        
        hypothesis = self.hypotheses[hypothesis_id]
        hypothesis.status = "testing"
        
        # Generate evidence for and against
        evidence_for = self._generate_supporting_evidence(hypothesis)
        evidence_against = self._generate_contradicting_evidence(hypothesis)
        
        # Update hypothesis with evidence
        hypothesis.evidence_for.extend(evidence_for)
        hypothesis.evidence_against.extend(evidence_against)
        
        # Calculate test result
        support_strength = len(evidence_for) * 0.2
        contradiction_strength = len(evidence_against) * 0.3
        
        # Update confidence
        confidence_change = support_strength - contradiction_strength
        hypothesis.confidence = max(0.0, min(1.0, hypothesis.confidence + confidence_change))
        
        # Determine status
        if hypothesis.confidence > 0.8 and len(evidence_for) > len(evidence_against):
            hypothesis.status = "confirmed"
        elif hypothesis.confidence < 0.3 or len(evidence_against) > len(evidence_for):
            hypothesis.status = "refuted"
        else:
            hypothesis.status = "inconclusive"
        
        test_result = {
            'hypothesis_id': hypothesis_id,
            'status': hypothesis.status,
            'confidence': hypothesis.confidence,
            'evidence_for': evidence_for,
            'evidence_against': evidence_against
        }
        
        hypothesis.test_results = test_result
        
        return test_result
    
    def _generate_supporting_evidence(self, hypothesis: Hypothesis) -> List[str]:
        """Generate evidence that supports the hypothesis"""
        # Simplified evidence generation based on hypothesis content
        statement_lower = hypothesis.statement.lower()
        evidence = []
        
        # Domain-specific evidence generation
        if any(word in statement_lower for word in ['math', 'calculate', 'number']):
            evidence.extend([
                "Mathematical principles support this approach",
                "Numerical analysis validates the method",
                "Calculations are consistent with known formulas"
            ])
        elif any(word in statement_lower for word in ['code', 'program', 'software']):
            evidence.extend([
                "Code structure follows best practices",
                "Algorithm efficiency is optimal",
                "Implementation is technically sound"
            ])
        elif any(word in statement_lower for word in ['design', 'create', 'build']):
            evidence.extend([
                "Design principles are well-established",
                "Similar approaches have been successful",
                "Requirements are clearly defined"
            ])
        else:
            evidence.extend([
                "Logic is sound and consistent",
                "Reasoning follows established patterns",
                "Conclusion is well-supported"
            ])
        
        # Return subset based on confidence
        num_evidence = min(len(evidence), int(hypothesis.confidence * 5))
        return evidence[:num_evidence]
    
    def _generate_contradicting_evidence(self, hypothesis: Hypothesis) -> List[str]:
        """Generate evidence that contradicts the hypothesis"""
        statement_lower = hypothesis.statement.lower()
        evidence = []
        
        # Generate potential counterarguments
        if any(word in statement_lower for word in ['always', 'never', 'all', 'none']):
            evidence.extend([
                "Absolute statements are rarely true",
                "Exceptions to general rules exist",
                "Edge cases may not be considered"
            ])
        
        if any(word in statement_lower for word in ['simple', 'easy', 'straightforward']):
            evidence.extend([
                "Complexity may be underestimated",
                "Hidden difficulties may exist",
                "Simplicity assumption may be flawed"
            ])
        
        if hypothesis.confidence > 0.9:
            evidence.extend([
                "High confidence may indicate overconfidence",
                "Alternative explanations should be considered",
                "More evidence is needed for such certainty"
            ])
        
        # Return subset based on inverse confidence
        num_evidence = min(len(evidence), int((1.0 - hypothesis.confidence) * 3))
        return evidence[:num_evidence]

class DeductionEngine:
    """Engine for multi-step logical deduction"""
    
    def __init__(self, config, concept_bank, consciousness_core):
        self.config = config
        self.concept_bank = concept_bank
        self.consciousness_core = consciousness_core
        
        # Deduction rules and patterns
        self.logical_rules = self._initialize_logical_rules()
        self.deduction_history = []
        
    def _initialize_logical_rules(self) -> Dict:
        """Initialize basic logical deduction rules"""
        return {
            'modus_ponens': {
                'pattern': ['If A then B', 'A'],
                'conclusion': 'B',
                'confidence': 0.95
            },
            'modus_tollens': {
                'pattern': ['If A then B', 'Not B'],
                'conclusion': 'Not A',
                'confidence': 0.95
            },
            'syllogism': {
                'pattern': ['All A are B', 'All B are C'],
                'conclusion': 'All A are C',
                'confidence': 0.9
            },
            'hypothetical_syllogism': {
                'pattern': ['If A then B', 'If B then C'],
                'conclusion': 'If A then C',
                'confidence': 0.85
            }
        }
    
    def perform_deduction(self, premises: List[str]) -> Dict:
        """Perform logical deduction from premises"""
        deduction_id = f"deduction_{uuid.uuid4().hex[:8]}"
        
        # Analyze premises
        analyzed_premises = [self._analyze_premise(p) for p in premises]
        
        # Apply logical rules
        deductions = []
        for rule_name, rule in self.logical_rules.items():
            result = self._apply_rule(rule, analyzed_premises)
            if result:
                deductions.append({
                    'rule': rule_name,
                    'conclusion': result['conclusion'],
                    'confidence': result['confidence'],
                    'premises_used': result['premises_used']
                })
        
        # Multi-step deduction
        if len(deductions) > 0:
            extended_deductions = self._extend_deductions(deductions, analyzed_premises)
            deductions.extend(extended_deductions)
        
        deduction_result = {
            'deduction_id': deduction_id,
            'premises': premises,
            'deductions': deductions,
            'timestamp': time.time()
        }
        
        self.deduction_history.append(deduction_result)
        
        return deduction_result
    
    def _analyze_premise(self, premise: str) -> Dict:
        """Analyze premise structure"""
        premise_lower = premise.lower().strip()
        
        analysis = {
            'original': premise,
            'processed': premise_lower,
            'type': 'unknown',
            'subject': None,
            'predicate': None,
            'condition': None
        }
        
        # Identify premise type
        if 'if' in premise_lower and 'then' in premise_lower:
            analysis['type'] = 'conditional'
            parts = premise_lower.split('then')
            if len(parts) == 2:
                analysis['condition'] = parts[0].replace('if', '').strip()
                analysis['predicate'] = parts[1].strip()
        elif 'all' in premise_lower and 'are' in premise_lower:
            analysis['type'] = 'universal'
            parts = premise_lower.split('are')
            if len(parts) == 2:
                analysis['subject'] = parts[0].replace('all', '').strip()
                analysis['predicate'] = parts[1].strip()
        elif 'not' in premise_lower:
            analysis['type'] = 'negation'
            analysis['predicate'] = premise_lower.replace('not', '').strip()
        else:
            analysis['type'] = 'assertion'
            analysis['predicate'] = premise_lower
        
        return analysis
    
    def _apply_rule(self, rule: Dict, premises: List[Dict]) -> Optional[Dict]:
        """Apply logical rule to premises"""
        rule_pattern = rule['pattern']
        
        # Simple pattern matching for demonstration
        if rule['pattern'] == ['If A then B', 'A']:
            # Look for conditional and its antecedent
            conditional = None
            antecedent = None
            
            for premise in premises:
                if premise['type'] == 'conditional':
                    conditional = premise
                elif premise['type'] == 'assertion' and conditional:
                    # Check if assertion matches condition
                    if premise['predicate'] in conditional['condition']:
                        antecedent = premise
                        break
            
            if conditional and antecedent:
                return {
                    'conclusion': conditional['predicate'],
                    'confidence': rule['confidence'],
                    'premises_used': [conditional['original'], antecedent['original']]
                }
        
        elif rule['pattern'] == ['All A are B', 'All B are C']:
            # Look for two universal statements that can be chained
            universals = [p for p in premises if p['type'] == 'universal']
            
            if len(universals) >= 2:
                for i, u1 in enumerate(universals):
                    for u2 in universals[i+1:]:
                        # Check if predicate of first matches subject of second
                        if u1['predicate'] and u2['subject'] and u1['predicate'] in u2['subject']:
                            return {
                                'conclusion': f"All {u1['subject']} are {u2['predicate']}",
                                'confidence': rule['confidence'],
                                'premises_used': [u1['original'], u2['original']]
                            }
        
        return None
    
    def _extend_deductions(self, initial_deductions: List[Dict], premises: List[Dict]) -> List[Dict]:
        """Extend deductions with multi-step reasoning"""
        extended = []
        
        # Add conclusions as new premises
        for deduction in initial_deductions:
            new_premise = {
                'original': deduction['conclusion'],
                'processed': deduction['conclusion'].lower(),
                'type': 'derived',
                'predicate': deduction['conclusion']
            }
            
            # Try to apply rules with new premise included
            extended_premises = premises + [new_premise]
            
            for rule_name, rule in self.logical_rules.items():
                result = self._apply_rule(rule, extended_premises)
                if result and result['conclusion'] not in [d['conclusion'] for d in initial_deductions]:
                    result['rule'] = rule_name
                    result['derived_from'] = deduction['conclusion']
                    extended.append(result)
        
        return extended

class ReasoningMemory:
    """Memory system for storing and retrieving reasoning patterns"""
    
    def __init__(self, config, concept_bank):
        self.config = config
        self.concept_bank = concept_bank
        
        # Memory storage
        self.stored_chains = {}  # chain_id -> ReasoningChain
        self.reasoning_patterns = defaultdict(list)
        self.successful_strategies = defaultdict(list)
        
        # Indexing for fast retrieval
        self.problem_type_index = defaultdict(list)
        self.confidence_index = defaultdict(list)
        
    def store_chain(self, chain: ReasoningChain):
        """Store completed reasoning chain"""
        self.stored_chains[chain.chain_id] = chain
        
        # Index by problem characteristics
        problem_type = self._classify_problem_type(chain.problem_statement)
        self.problem_type_index[problem_type].append(chain.chain_id)
        
        # Index by confidence level
        confidence_bucket = int(chain.confidence_score * 10) / 10
        self.confidence_index[confidence_bucket].append(chain.chain_id)
        
        # Extract and store patterns
        self._extract_patterns(chain)
        
        # Store successful strategies
        if chain.confidence_score > 0.7:
            strategy = self._extract_strategy(chain)
            self.successful_strategies[problem_type].append(strategy)
    
    def _classify_problem_type(self, problem: str) -> str:
        """Classify problem into broad categories"""
        problem_lower = problem.lower()
        
        if any(word in problem_lower for word in ['calculate', 'compute', 'math', 'number']):
            return 'mathematical'
        elif any(word in problem_lower for word in ['code', 'program', 'algorithm', 'software']):
            return 'programming'
        elif any(word in problem_lower for word in ['design', 'create', 'build', 'develop']):
            return 'creative'
        elif any(word in problem_lower for word in ['analyze', 'understand', 'explain', 'interpret']):
            return 'analytical'
        elif any(word in problem_lower for word in ['plan', 'strategy', 'organize', 'schedule']):
            return 'planning'
        else:
            return 'general'
    
    def _extract_patterns(self, chain: ReasoningChain):
        """Extract reasoning patterns from chain"""
        if len(chain.steps) < 2:
            return
        
        # Extract step transition patterns
        for i in range(len(chain.steps) - 1):
            current_step = chain.steps[i]
            next_step = chain.steps[i + 1]
            
            pattern = {
                'from_type': current_step.step_type.value,
                'to_type': next_step.step_type.value,
                'confidence_change': next_step.confidence - current_step.confidence,
                'consciousness_level': current_step.consciousness_level,
                'problem_type': self._classify_problem_type(chain.problem_statement)
            }
            
            pattern_key = f"{current_step.step_type.value}->{next_step.step_type.value}"
            self.reasoning_patterns[pattern_key].append(pattern)
    
    def _extract_strategy(self, chain: ReasoningChain) -> Dict:
        """Extract successful reasoning strategy"""
        step_sequence = [step.step_type.value for step in chain.steps]
        
        return {
            'step_sequence': step_sequence,
            'total_steps': len(chain.steps),
            'confidence_score': chain.confidence_score,
            'problem_type': self._classify_problem_type(chain.problem_statement),
            'avg_consciousness': np.mean([step.consciousness_level for step in chain.steps])
        }
    
    def retrieve_similar_chains(self, problem: str, top_k: int = 5) -> List[ReasoningChain]:
        """Retrieve similar reasoning chains"""
        problem_type = self._classify_problem_type(problem)
        
        # Get chains of same problem type
        candidate_chains = []
        for chain_id in self.problem_type_index.get(problem_type, []):
            if chain_id in self.stored_chains:
                candidate_chains.append(self.stored_chains[chain_id])
        
        # If not enough, get from other types
        if len(candidate_chains) < top_k:
            for other_type, chain_ids in self.problem_type_index.items():
                if other_type != problem_type:
                    for chain_id in chain_ids:
                        if chain_id in self.stored_chains:
                            candidate_chains.append(self.stored_chains[chain_id])
                            if len(candidate_chains) >= top_k * 2:
                                break
        
        # Sort by confidence and recency
        candidate_chains.sort(
            key=lambda c: (c.confidence_score, c.created_at),
            reverse=True
        )
        
        return candidate_chains[:top_k]
    
    def get_successful_strategies(self, problem_type: str) -> List[Dict]:
        """Get successful strategies for problem type"""
        strategies = self.successful_strategies.get(problem_type, [])
        
        # Sort by confidence score
        strategies.sort(key=lambda s: s['confidence_score'], reverse=True)
        
        return strategies[:5]  # Top 5 strategies

# Integration with existing SAM architecture
class ReasoningIntegration:
    """Integration layer for reasoning systems with SAM"""
    
    def __init__(self, sam_model):
        self.sam = sam_model
        self.config = sam_model.config
        
        # Initialize reasoning systems
        self.cot_processor = ChainOfThoughtProcessor(
            self.config, 
            sam_model.concept_bank, 
            sam_model.consciousness
        )
        
        self.got_planner = GraphOfThoughtPlanner(
            self.config,
            sam_model.concept_bank,
            sam_model.consciousness,
            self.cot_processor
        )
        
        # Add to SAM model
        sam_model.reasoning = self
        sam_model.chain_of_thought = self.cot_processor
        sam_model.graph_of_thought = self.got_planner
        
        logger.info("Reasoning systems integrated with SAM")
    
    def solve_problem(self, problem: str, use_graph_planning: bool = True) -> Dict:
        """Solve problem using chain-of-thought and graph-of-thought"""
        start_time = time.time()
        
        # Get consciousness context
        consciousness_context = self.sam.consciousness.get_consciousness_context()
        
        # Create initial reasoning chain
        chain_id = self.cot_processor.create_reasoning_chain(problem, consciousness_context)
        
        # For complex problems, use graph planning
        if use_graph_planning and len(problem.split()) > 10:
            # Create planning graph
            graph_id = self.got_planner.create_planning_graph(problem)
            
            # Add hypothesis nodes for potential solutions
            root_hypotheses = self._generate_initial_hypotheses(problem)
            hypothesis_nodes = []
            
            for hypothesis in root_hypotheses:
                hyp_node = self.got_planner.add_hypothesis_node(
                    graph_id, "root", hypothesis
                )
                hypothesis_nodes.append(hyp_node)
            
            # Test hypotheses
            test_results = []
            for hyp_node in hypothesis_nodes:
                result = self.got_planner.test_hypothesis_node(graph_id, hyp_node)
                test_results.append(result)
                
                # Add test results to reasoning chain
                self.cot_processor.add_reasoning_step(
                    chain_id,
                    f"Tested hypothesis: {result.get('status', 'unknown')}",
                    ReasoningStep.VERIFICATION
                )
            
            # Execute plan
            execution_result = self.got_planner.execute_plan(graph_id)
            
            # Add final conclusion to chain
            if execution_result.get('success'):
                self.cot_processor.add_reasoning_step(
                    chain_id,
                    f"Solution: {execution_result.get('final_answer', 'Plan executed successfully')}",
                    ReasoningStep.CONCLUSION
                )
            
            solution = {
                'problem': problem,
                'solution_method': 'graph_of_thought',
                'chain_id': chain_id,
                'graph_id': graph_id,
                'reasoning_chain': self.cot_processor.get_chain_summary(chain_id),
                'planning_graph': self.got_planner.get_graph_visualization(graph_id),
                'execution_result': execution_result,
                'processing_time': time.time() - start_time
            }
            
        else:
            # Use pure chain-of-thought reasoning
            # Add reasoning steps
            self._add_analytical_steps(chain_id, problem)
            
            # Get final solution
            chain_summary = self.cot_processor.get_chain_summary(chain_id)
            
            solution = {
                'problem': problem,
                'solution_method': 'chain_of_thought',
                'chain_id': chain_id,
                'reasoning_chain': chain_summary,
                'final_answer': chain_summary.get('final_answer'),
                'processing_time': time.time() - start_time
            }
        
        return solution
    
    def _generate_initial_hypotheses(self, problem: str) -> List[str]:
        """Generate initial hypotheses for problem"""
        problem_lower = problem.lower()
        hypotheses = []
        
        # Problem-type specific hypotheses
        if any(word in problem_lower for word in ['calculate', 'compute', 'solve']):
            hypotheses.extend([
                "This requires mathematical calculation",
                "A formula or equation should be applied",
                "Numerical methods may be needed"
            ])
        elif any(word in problem_lower for word in ['design', 'create', 'build']):
            hypotheses.extend([
                "This requires creative problem solving",
                "Multiple design approaches should be considered",
                "Requirements need to be clearly defined"
            ])
        elif any(word in problem_lower for word in ['analyze', 'understand']):
            hypotheses.extend([
                "This requires analytical reasoning",
                "Information needs to be gathered and processed",
                "Patterns or relationships should be identified"
            ])
        else:
            hypotheses.extend([
                "This problem has multiple potential solutions",
                "A systematic approach is needed",
                "Breaking down into subproblems would help"
            ])
        
        return hypotheses[:3]  # Return top 3 hypotheses
    
    def _add_analytical_steps(self, chain_id: str, problem: str):
        """Add analytical reasoning steps to chain"""
        # Step 1: Problem analysis
        self.cot_processor.add_reasoning_step(
            chain_id,
            f"Analyzing the problem: {problem}",
            ReasoningStep.OBSERVATION
        )
        
        # Step 2: Approach consideration
        self.cot_processor.add_reasoning_step(
            chain_id,
            "Considering different approaches to solve this problem",
            ReasoningStep.HYPOTHESIS
        )
        
        # Step 3: Reasoning through solution
        self.cot_processor.add_reasoning_step(
            chain_id,
            "Working through the logical steps to reach a solution",
            ReasoningStep.DEDUCTION
        )
        
        # Step 4: Solution
        self.cot_processor.add_reasoning_step(
            chain_id,
            f"Based on the analysis, the solution is to systematically address: {problem}",
            ReasoningStep.CONCLUSION
        )
    
    def get_reasoning_stats(self) -> Dict:
        """Get comprehensive reasoning statistics"""
        return {
            'chain_of_thought': self.cot_processor.reasoning_stats,
            'graph_of_thought': self.got_planner.planning_stats,
            'active_chains': len(self.cot_processor.active_chains),
            'active_graphs': len(self.got_planner.active_graphs),
            'reasoning_patterns': len(self.cot_processor.reasoning_patterns)
        }

# Enhanced SAM forward method integration
def enhance_sam_with_reasoning(sam_model):
    """Enhance existing SAM model with reasoning capabilities"""
    
    # Add reasoning integration
    reasoning_integration = ReasoningIntegration(sam_model)
    
    # Store original forward method
    original_forward = sam_model.forward
    
    def enhanced_forward(self, input_text=None, input_concepts=None, target_concepts=None,
                        attention_mask=None, modality="text", return_dict=False,
                        enable_reasoning=False, reasoning_strategy="auto"):
        """Enhanced forward with reasoning capabilities"""
        
        # Call original forward
        result = original_forward(input_text, input_concepts, target_concepts,
                                attention_mask, modality, return_dict)
        
        # Add reasoning if enabled
        if enable_reasoning and input_text:
            reasoning_result = reasoning_integration.solve_problem(
                input_text, 
                use_graph_planning=(reasoning_strategy in ["graph", "auto"] and len(input_text.split()) > 10)
            )
            
            if return_dict:
                if isinstance(result, dict):
                    result['reasoning'] = reasoning_result
                else:
                    result = {'model_output': result, 'reasoning': reasoning_result}
        
        return result
    
    # Replace forward method
    sam_model.forward = enhanced_forward.__get__(sam_model, sam_model.__class__)
    
    logger.info("SAM enhanced with reasoning capabilities")
    return sam_model

# Usage example integration
def create_reasoning_enabled_sam(config=None):
    """Create SAM with reasoning capabilities enabled"""
    from sam import SAM, SAMConfig
    
    # Create base SAM
    if config is None:
        config = SAMConfig()
    
    sam = SAM(config)
    
    # Enhance with reasoning
    enhanced_sam = enhance_sam_with_reasoning(sam)
    
    return enhanced_sam
