"""Core data types for LDSG v2.0."""
import uuid
import time
import numpy as np
from encoder import encode

class Node:
    """A node in the spatial graph."""
    def __init__(self, concept, vector, layer='L2'):
        self.id = str(uuid.uuid4())[:8]
        self.concept = concept
        self.vector = np.asarray(vector).flatten()  # always 1D
        self.layer = layer    # L1=subnode, L2=key, L3=meta
        self.edges = {}       # {target_id: (weight, relation, context_vec)}
        self.subnode_bundle = []  # list of SubnodeSignature for L1
        self.position = self.vector.copy()  # spatial position in graph

    def add_edge(self, target_id, weight, relation, context_vec=None):
        self.edges[target_id] = (weight, relation, context_vec or np.zeros(128))

    def get_subnode_bundle_summary(self):
        """Return summary vector of subnode bundle."""
        if not self.subnode_bundle:
            dim = len(self.subnode_bundle[0].vector) if self.subnode_bundle else 128
            return np.zeros(dim)
        vectors = [np.asarray(s.vector).flatten() for s in self.subnode_bundle]
        return np.mean(vectors, axis=0)

class SubnodeSignature:
    """Compressed signature for a subnode (L1 layer)."""
    def __init__(self, concept, vector, role, activation=1.0):
        self.concept = concept
        self.vector = vector  # 64-dim or 128-dim compressed
        self.role = role      # challenge, stakeholder, symptom, etc.
        self.activation = activation

class ShortTermSubgraph:
    """Short-term memory subgraph for one conversation session."""
    def __init__(self, session_id):
        self.session_id = session_id
        self.key_nodes = []   # list of Node objects
        self.sub_nodes = []   # list of dicts {concept, role, activation}
        self.edges = []       # list of (src_id, tgt_id, weight, relation)

    def add_key_node(self, concept, vector):
        """Add a key node to STM."""
        node = Node(concept, vector, layer='L2')
        self.key_nodes.append(node)
        return node

    def add_relation(self, src_id, tgt_id, relation, weight=1.0):
        """Add a relation between key nodes."""
        self.edges.append((src_id, tgt_id, weight, relation))

    def get_all_node_ids(self):
        return [n.id for n in self.key_nodes]


# =============================================================================
# v2.0: 子图结构
# =============================================================================


class Subgraph:
    """
    一个独立的子图，对应一个任务产生的记忆块。

    初始状态与主图物理隔离（isolated=True），融入主图后 isolated=False。
    """

    def __init__(
        self,
        id: str | None = None,
        task_id: str | None = None,
        nodes: list[str] | None = None,
        edges: list[tuple] | None = None,
        weight: float = 1.0,
        isolated: bool = True,
    ):
        self.id = id or str(uuid.uuid4())
        self.task_id = task_id or ""
        self.nodes: list[str] = nodes or []       # 节点ID列表
        self.edges: list[tuple] = edges or []      # (src, tgt, weight) 元组列表
        self.weight = weight                       # 与主图融合程度
        self.isolated = isolated                   # 是否与主图隔离
        self.created_at = time.time()
        self.last_activated = time.time()

    def add_nodes(self, node_ids: list[str]):
        """添加节点"""
        for n in node_ids:
            if n not in self.nodes:
                self.nodes.append(n)

    def add_edge(self, src: str, tgt: str, weight: float = 1.0):
        """添加边"""
        self.edges.append((src, tgt, weight))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "nodes": self.nodes,
            "edges": self.edges,
            "weight": self.weight,
            "isolated": self.isolated,
            "created_at": self.created_at,
            "last_activated": self.last_activated,
        }

    @staticmethod
    def from_dict(d: dict) -> "Subgraph":
        sg = Subgraph(
            id=d["id"],
            task_id=d.get("task_id", ""),
            nodes=d.get("nodes", []),
            edges=d.get("edges", []),
            weight=d.get("weight", 1.0),
            isolated=d.get("isolated", True),
        )
        sg.created_at = d.get("created_at", time.time())
        sg.last_activated = d.get("last_activated", time.time())
        return sg

