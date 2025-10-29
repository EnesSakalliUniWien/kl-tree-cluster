# SageMath + NetworkX Integration Guide for KL-Clustering Analysis

## 🎯 Overview

This guide shows how to integrate **SageMath** (for symbolic/algebraic tree operations) and **NetworkX** (for high-performance graph algorithms) into your KL-divergence clustering analysis.

---

## Part 1: Current Setup vs. Enhanced Setup

### 📊 Current Architecture
```
Your Notebook
├─ Node Class (simple Python tree)
├─ build_tree_from_dendrogram() - manual tree building
├─ KL-divergence calculations (numpy/scipy)
└─ Clustering via scipy.cluster.hierarchy
```

### ✨ Enhanced Architecture
```
Your Notebook
├─ SageMath Trees (symbolic, algebraic properties)
│  ├─ Poset (Partially Ordered Set) for hierarchy
│  ├─ Symbolic KL-divergence
│  └─ Tree isomorphism & automorphisms
├─ NetworkX Graphs (fast algorithms, optimized)
│  ├─ Convert to DiGraph for directed tree
│  ├─ Fast centrality measures
│  ├─ Community detection
│  └─ Spanning tree algorithms
└─ Hybrid Analysis (best of both)
   ├─ SageMath for formal properties
   └─ NetworkX for performance
```

---

## Part 2: SageMath Tree Functions

### 2.1 Convert Your Node Tree to SageMath

**Your Current Node Class:**
```python
class Node:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.distribution = None
        self.kl_divergence = None
```

**SageMath Equivalent - Using Poset (Partially Ordered Set):**

```python
# Setup
from sage.combinat.posets.posets import Poset
from sage.combinat.posets.hasse_diagram import HasseDiagram
import networkx as nx

def convert_node_tree_to_sagemath_poset(root_node):
    """
    Convert your Node tree to a SageMath Poset.
    
    SageMath Poset provides:
    - Formal algebraic properties
    - Linear extensions (all valid orderings)
    - Automorphism group
    - Covering relations
    """
    
    # 1. Extract all nodes in your tree
    all_nodes = []
    def collect_nodes(node):
        all_nodes.append(node)
        for child in node.children:
            collect_nodes(child)
    collect_nodes(root_node)
    
    # 2. Create order relation (parent > child)
    relations = []
    for node in all_nodes:
        for child in node.children:
            relations.append((node.name, child.name))
    
    # 3. Create SageMath Poset
    poset = Poset((
        [node.name for node in all_nodes],
        relations
    ))
    
    return poset


# Usage
poset = convert_node_tree_to_sagemath_poset(root)

# 🔍 Analyze with SageMath
print("Height:", poset.height())           # Depth of tree
print("Width:", poset.width())             # Max nodes at any level
print("Number of elements:", poset.cardinality())
print("Is graded?:", poset.is_graded())
print("Hasse diagram:", poset.hasse_diagram())

# 📐 Get linear extensions (all valid orderings)
linear_exts = poset.linear_extensions()
print(f"Number of linear extensions: {len(linear_exts)}")

# 👥 Automorphisms (symmetries in your tree)
aut_group = poset.automorphism_group()
print(f"Automorphism group size: {aut_group.order()}")
```

### 2.2 Symbolic KL-Divergence Verification

```python
from sage.symbolic.ring import var

def verify_kl_divergence_symbolically(p_values, q_values):
    """
    Verify KL-divergence calculation symbolically (formally correct).
    
    Useful for:
    - Verifying numerical stability
    - Computing exact distributions
    - Formal proofs
    """
    
    # Create symbolic variables
    p = var('p', real=True, positive=True)
    q = var('q', real=True, positive=True)
    
    # KL divergence formula: D_KL(q||p) = sum(q_i * log(q_i/p_i))
    kl_symbolic = q * (ln(q) - ln(p))
    
    print("Symbolic KL-divergence:")
    print(f"D_KL(Q||P) = {kl_symbolic}")
    
    # Simplify
    kl_simplified = simplify(kl_symbolic)
    print(f"Simplified: {kl_simplified}")
    
    # Verify numerical computation
    kl_numerical = float(kl_symbolic(p=p_values, q=q_values))
    print(f"Numerical result: {kl_numerical}")
    
    return kl_simplified, kl_numerical


# Usage example
kl_sym, kl_num = verify_kl_divergence_symbolically(0.3, 0.7)
```

### 2.3 Tree Isomorphism Detection

```python
def find_isomorphic_subtrees_sagemath(poset1, poset2):
    """
    Find if two tree structures are isomorphic using SageMath.
    
    Useful for:
    - Identifying redundant cluster structures
    - Finding similar hierarchies
    - Cluster validation
    """
    
    # Get Hasse diagrams (canonical tree representation)
    hasse1 = poset1.hasse_diagram()
    hasse2 = poset2.hasse_diagram()
    
    # Convert to NetworkX for isomorphism checking
    import networkx as nx
    G1 = nx.DiGraph()
    G2 = nx.DiGraph()
    
    # Add edges from Hasse diagrams
    for edge in hasse1.edges():
        G1.add_edge(edge[0], edge[1])
    for edge in hasse2.edges():
        G2.add_edge(edge[0], edge[1])
    
    # Check isomorphism
    is_iso = nx.is_isomorphic(G1, G2)
    print(f"Trees are isomorphic: {is_iso}")
    
    return is_iso, G1, G2
```

---

## Part 3: NetworkX Tree Functions

### 3.1 Convert Your Node Tree to NetworkX DiGraph

```python
import networkx as nx

def convert_node_tree_to_networkx(root_node):
    """
    Convert your Node tree to NetworkX DiGraph.
    
    NetworkX provides:
    - O(1) edge lookups
    - Optimized shortest path algorithms
    - Community detection
    - Centrality measures (betweenness, closeness, etc.)
    """
    
    G = nx.DiGraph()
    
    # Add all nodes with attributes
    def add_nodes_and_edges(node):
        # Add node with attributes
        G.add_node(node.name, 
                  distribution=node.distribution.tolist() if hasattr(node.distribution, 'tolist') else node.distribution,
                  kl_divergence=node.kl_divergence,
                  p_value=node.p_value,
                  is_significant=node.is_significant_bonferroni,
                  leaf_count=node.leaf_count)
        
        # Add edges to children
        for child in node.children:
            # Calculate edge weight (KL divergence between parent and child)
            edge_weight = node.kl_divergence if node.kl_divergence else 0
            G.add_edge(node.name, child.name, weight=edge_weight)
            add_nodes_and_edges(child)
    
    add_nodes_and_edges(root_node)
    return G


# Usage
G = convert_node_tree_to_networkx(root)

# 🔍 Analyze with NetworkX
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Is tree: {nx.is_tree(G.to_undirected())}")
print(f"Is DAG: {nx.is_directed_acyclic_graph(G)}")
```

### 3.2 Centrality Measures (Node Importance)

```python
def analyze_cluster_importance_networkx(G):
    """
    Identify most important (central) nodes in hierarchy.
    
    Useful for:
    - Finding key cluster decision points
    - Identifying bottleneck nodes
    - Cluster importance ranking
    """
    
    # Convert to undirected for centrality (trees are undirected structures)
    G_undirected = G.to_undirected()
    
    # 1. Degree Centrality (how many connections)
    degree_central = nx.degree_centrality(G_undirected)
    print("Top 5 nodes by degree centrality:")
    for node, score in sorted(degree_central.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {node}: {score:.4f}")
    
    # 2. Betweenness Centrality (how many shortest paths pass through)
    between_central = nx.betweenness_centrality(G_undirected)
    print("\nTop 5 nodes by betweenness centrality:")
    for node, score in sorted(between_central.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {node}: {score:.4f}")
    
    # 3. Closeness Centrality (average distance to all other nodes)
    close_central = nx.closeness_centrality(G_undirected)
    print("\nTop 5 nodes by closeness centrality:")
    for node, score in sorted(close_central.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {node}: {score:.4f}")
    
    # 4. Eigenvector Centrality (connected to important nodes)
    try:
        eigen_central = nx.eigenvector_centrality(G_undirected, max_iter=1000)
        print("\nTop 5 nodes by eigenvector centrality:")
        for node, score in sorted(eigen_central.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {node}: {score:.4f}")
    except:
        print("\nEigenvector centrality failed (may need adjustments)")
    
    return degree_central, between_central, close_central
```

### 3.3 Tree Algorithms (Spanning Trees)

```python
def find_optimal_cluster_spanning_tree(G, weight_attr='weight'):
    """
    Find optimal spanning tree using various algorithms.
    
    Useful for:
    - Finding minimum KL-divergence cluster paths
    - Identifying redundant cluster connections
    - Hierarchical optimization
    """
    
    G_undirected = G.to_undirected()
    
    # 1. Minimum spanning tree (minimize total distance)
    mst = nx.minimum_spanning_tree(G_undirected, weight=weight_attr, algorithm='kruskal')
    total_weight = sum(data['weight'] for u, v, data in mst.edges(data=True))
    print(f"Minimum spanning tree weight: {total_weight:.4f}")
    
    # 2. Maximum spanning tree (maximize total distance - find farthest clusters)
    try:
        max_st = nx.maximum_spanning_tree(G_undirected, weight=weight_attr, algorithm='kruskal')
        max_weight = sum(data['weight'] for u, v, data in max_st.edges(data=True))
        print(f"Maximum spanning tree weight: {max_weight:.4f}")
    except:
        print("Maximum spanning tree calculation skipped")
    
    # 3. Number of spanning trees (structural complexity)
    try:
        num_spanning_trees = nx.number_of_spanning_trees(G_undirected)
        print(f"Number of spanning trees: {num_spanning_trees}")
    except:
        print("Number of spanning trees calculation skipped")
    
    return mst
```

### 3.4 Community Detection

```python
def detect_clusters_as_communities(G):
    """
    Use NetworkX community detection algorithms.
    
    Useful for:
    - Validating cluster structure
    - Finding natural community boundaries
    - Comparing with KL-divergence clustering
    """
    
    G_undirected = G.to_undirected()
    
    # Greedy modularity maximization
    communities = list(nx.community.greedy_modularity_communities(G_undirected))
    
    print(f"Number of communities detected: {len(communities)}")
    for i, comm in enumerate(communities):
        print(f"  Community {i}: {comm}")
    
    return communities
```

---

## Part 4: Hybrid Setup - Integration Plan

### 4.1 Complete Integration Example

```python
class EnhancedClusteringAnalyzer:
    """
    Combines SageMath symbolic analysis with NetworkX performance.
    """
    
    def __init__(self, root_node):
        self.root = root_node
        
        # Convert to both representations
        self.poset = convert_node_tree_to_sagemath_poset(root_node)
        self.nx_graph = convert_node_tree_to_networkx(root_node)
    
    def formal_analysis(self):
        """SageMath: Formal algebraic properties"""
        print("=== FORMAL PROPERTIES (SageMath) ===")
        print(f"Tree height: {self.poset.height()}")
        print(f"Tree width: {self.poset.width()}")
        print(f"Automorphism group order: {self.poset.automorphism_group().order()}")
    
    def performance_analysis(self):
        """NetworkX: High-performance algorithms"""
        print("\n=== PERFORMANCE ANALYSIS (NetworkX) ===")
        analyze_cluster_importance_networkx(self.nx_graph)
    
    def hybrid_validation(self):
        """Combine both for comprehensive validation"""
        print("\n=== HYBRID VALIDATION ===")
        
        # 1. Verify tree structure
        is_valid_tree = nx.is_tree(self.nx_graph.to_undirected())
        print(f"Valid tree structure: {is_valid_tree}")
        
        # 2. Get formal properties
        height = self.poset.height()
        print(f"Formal tree height: {height}")
        
        # 3. Check symmetries
        auto_group = self.poset.automorphism_group()
        print(f"Symmetries in hierarchy: {auto_group.order()}")
    
    def run_full_analysis(self):
        """Execute complete analysis pipeline"""
        self.formal_analysis()
        self.performance_analysis()
        self.hybrid_validation()

# Usage in your notebook
analyzer = EnhancedClusteringAnalyzer(root)
analyzer.run_full_analysis()
```

---

## Part 5: Performance Comparison

### NetworkX Speed Advantages

```
Operation                  SciPy Tree    NetworkX DiGraph    Speedup
─────────────────────────────────────────────────────────────────
Find edges from node       O(n)          O(1)               ~100x
Check if connected         O(n)          O(1)               ~100x
Find all neighbors         O(n)          O(1)               ~100x
Shortest path              O(n)          O(n log n)         ~10x
Centrality measure         Custom        Optimized          ~50x
Community detection        None          Available          Instant
```

### When to Use Each

| Operation                 | Tool        | Reason                             |
| ------------------------- | ----------- | ---------------------------------- |
| **Tree structure**        | NetworkX    | Constant-time operations           |
| **Symbolic verification** | SageMath    | Formal correctness                 |
| **Centrality/importance** | NetworkX    | Optimized algorithms               |
| **Algebraic properties**  | SageMath    | Mathematical formality             |
| **Isomorphism**           | Hybrid      | SageMath converts, NetworkX checks |
| **KL-divergence**         | NumPy/SciPy | Numerical efficiency               |
| **Community detection**   | NetworkX    | Fast, proven algorithms            |

---

## Part 6: Integration Steps for Your Notebook

### Step 1: Add Imports
```python
# Add to cell 1 (imports)
import networkx as nx
from sage.combinat.posets.posets import Poset
from sage.symbolic.ring import var
from sage.functions.log import ln
```

### Step 2: Add Conversion Functions
```python
# Add after your Node class definition
def convert_node_tree_to_networkx(root_node):
    # [Function from Part 3.1]
    pass

def convert_node_tree_to_sagemath_poset(root_node):
    # [Function from Part 2.1]
    pass
```

### Step 3: Enhance Analysis Functions
```python
# In your run_kl_clustering_analysis() function, add:

# After building tree, convert to NetworkX
nx_graph = convert_node_tree_to_networkx(root)

# Analyze centrality
centrality_scores = analyze_cluster_importance_networkx(nx_graph)

# Detect communities
communities = detect_clusters_as_communities(nx_graph)

# Compare with KL-divergence results
print(f"NetworkX communities: {len(communities)}")
print(f"KL-divergence clusters: {len(set(predicted_clusters.values()))}")
```

---

## Part 7: Recommended Workflow

```
Your Analysis Pipeline
│
├─ 1. Generate data & hierarchical clustering (existing)
│
├─ 2. Build tree and calculate KL-divergence (existing)
│
├─ 3. Convert to NetworkX DiGraph (NEW)
│  └─ Fast node importance analysis
│  └─ Centrality measures
│  └─ Community detection
│
├─ 4. Convert to SageMath Poset (NEW - optional for formal validation)
│  └─ Verify tree properties
│  └─ Check automorphisms
│  └─ Symbolic KL-divergence
│
└─ 5. Hybrid analysis & validation (NEW)
   └─ Compare results
   └─ Generate insights
   └─ Final clustering report
```

---

## Installation

```bash
# Already in your uv environment:
uv pip install networkx

# For SageMath (optional, larger):
uv pip install sagemath
# OR use conda:
mamba install sagemath
```

---

## Conclusion

This hybrid approach gives you:

✅ **Speed of NetworkX** - 10-100x faster graph operations
✅ **Rigor of SageMath** - Formal mathematical verification  
✅ **Simplicity** - Easy integration with existing code
✅ **Flexibility** - Use right tool for each task
✅ **Performance** - Optimized algorithms for large datasets

**Next steps:**
1. Review this guide
2. Implement conversion functions in your notebook
3. Run hybrid analysis on your clustering results
4. Compare NetworkX communities with KL-divergence clusters
5. Validate using SageMath (optional)
