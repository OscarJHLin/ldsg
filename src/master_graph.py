"""Master Spatial Graph - the long-term memory layer."""
import json
import os
import time
import numpy as np
from core_types import Node, SubnodeSignature, ShortTermSubgraph, Subgraph
from encoder import encode, encode_flat

# Threshold for considering concepts "similar enough" to merge
# Lowered to avoid substring false merges (e.g. 项目A → 项目)
MERGE_THRESHOLD = 0.75  # cosine similarity

class MasterSpatialGraph:
    """Persistent long-term memory using spatial graph representation."""

    def _find_similar_node(self, vector, layer='L2'):
        """Find most similar existing node. Returns (node_id, similarity) or (None, sim)."""
        vec = np.asarray(vector).flatten()
        best_sim = -1
        best_id = None
        for nid, node in self.nodes.items():
            if node.layer == layer:
                node_vec = np.asarray(node.vector).flatten()
                sim = float(np.dot(vec, node_vec))
                if sim > best_sim:
                    best_sim = sim
                    best_id = nid
        if best_sim > MERGE_THRESHOLD:
            return best_id, best_sim
        return None, best_sim

    def add_or_update_node(self, concept, vector=None):
        """Add a new node or find existing similar one."""
        if vector is None:
            vector = encode_flat(concept)
        vec = np.asarray(vector).flatten()
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        if self.dim is None:
            self.dim = len(vec)

        # Find similar existing node
        existing_id, sim = self._find_similar_node(vec)

        # Safety guard: don't merge if concept strings suggest different entities
        if existing_id is not None:
            existing_concept = self.nodes[existing_id].concept
            # Don't merge if one is a prefix/suffix/substring of the other
            if (concept.startswith(existing_concept) or existing_concept.startswith(concept) or
                concept in existing_concept or existing_concept in concept):
                if concept != existing_concept:
                    existing_id = None  # force create new node

        if existing_id is not None:
            return self.nodes[existing_id]

        node = Node(concept, vec, layer='L2')
        self.nodes[node.id] = node
        return node

    def merge_short_term_memory(self, stm, encode_cache=None):
        """Merge STM subgraph into MSG. encode_cache: dict mapping concept->vector to ensure consistent encoding."""
        if encode_cache is None:
            encode_cache = {}
        mapping = {}  # stm_id -> msg_id
        stm_node_map = {}  # stm_id -> (is_new, msg_node_id or concept_for_new)

        # Step 1: Map key nodes
        for stm_node in stm.key_nodes:
            if stm_node.id in encode_cache:
                vec = encode_cache[stm_node.id]
            else:
                vec = stm_node.vector
            existing, sim = self._find_similar_node(vec)

            if existing:
                ec = self.nodes[existing].concept
                is_substring = (stm_node.concept.startswith(ec) or ec.startswith(stm_node.concept) or
                               stm_node.concept in ec or ec in stm_node.concept)
                if stm_node.concept != ec and is_substring:
                    existing = None  # guard: don't merge substring-like

            if existing:
                # Merge: update position slightly
                mapping[stm_node.id] = existing
                stm_node_map[stm_node.id] = (False, existing)
                old_vec = np.asarray(self.nodes[existing].vector).flatten()
                new_vec = 0.7 * old_vec + 0.3 * np.asarray(vec).flatten()
                new_vec = new_vec / (np.linalg.norm(new_vec) + 1e-8)
                self.nodes[existing].vector = new_vec
                self.nodes[existing].position = new_vec.copy()
            else:
                # Create new
                v = np.asarray(vec).flatten()
                v = v / (np.linalg.norm(v) + 1e-8)
                new_node = Node(stm_node.concept, v, layer='L2')
                self.nodes[new_node.id] = new_node
                mapping[stm_node.id] = new_node.id
                stm_node_map[stm_node.id] = (True, new_node.id)

        # Step 2: Create edges
        for knode in stm.key_nodes:
            for tgt_id, (weight, relation, _) in knode.edges.items():
                if knode.id in mapping and tgt_id in mapping:
                    msg_src = mapping[knode.id]
                    msg_tgt = mapping[tgt_id]
                    if msg_src != msg_tgt:
                        self.nodes[msg_src].add_edge(msg_tgt, weight, relation)

        # Step 3: Attach sub-nodes to primary newly-created key node
        newly_created_ids = [sid for sid, (is_new, _) in stm_node_map.items() if is_new]
        if not newly_created_ids:
            self.relax_layout(iterations=30, nodes=list(mapping.values()), verbose=False)
            return mapping

        # Find primary: most connected newly created node
        primary_stm = None
        max_edges = -1
        for sid in newly_created_ids:
            stm_node = next(n for n in stm.key_nodes if n.id == sid)
            if len(stm_node.edges) > max_edges:
                primary_stm = stm_node
                max_edges = len(stm_node.edges)

        if primary_stm and stm.sub_nodes and primary_stm.id in mapping:
            host_id = mapping[primary_stm.id]
            host = self.nodes[host_id]
            for sub_dict in stm.sub_nodes:
                sub_vec = encode_flat(sub_dict['concept'])
                sub_vec = sub_vec / (np.linalg.norm(sub_vec) + 1e-8)
                sub_sig = SubnodeSignature(
                    concept=sub_dict['concept'],
                    vector=sub_vec,
                    role=sub_dict['role'],
                    activation=sub_dict['activation']
                )
                host.subnode_bundle.append(sub_sig)

        # Step 4: Layout
        self.relax_layout(iterations=30, nodes=list(mapping.values()), verbose=False)
        return mapping

    def relax_layout(self, iterations=100, nodes=None, verbose=False):
        """Force-directed layout relaxation."""
        if not self.nodes:
            return

        if nodes is None:
            node_list = list(self.nodes.values())
        else:
            node_list = [self.nodes[nid] for nid in nodes if nid in self.nodes]

        if not node_list:
            return

        if self.dim is None and node_list:
            self.dim = len(node_list[0].position)

        # Initialize positions from vectors if not set
        for node in node_list:
            pos = np.asarray(node.position).flatten()
            if np.linalg.norm(pos) < 1e-6:
                pos = np.random.randn(self.dim)
            pos = pos / (np.linalg.norm(pos) + 1e-8)
            node.position = pos

        spring_k = 0.1
        repulsion_k = 0.01
        step_size = 0.01
        node_map = {n.id: i for i, n in enumerate(node_list)}

        for it in range(iterations):
            forces = np.zeros((len(node_list), self.dim))

            # Spring forces
            for node in node_list:
                i = node_map[node.id]
                for tgt_id, (weight, _, _) in node.edges.items():
                    if tgt_id not in node_map:
                        continue
                    j = node_map[tgt_id]
                    tgt_pos = node_list[j].position
                    diff = tgt_pos - node.position
                    dist = np.linalg.norm(diff)
                    if dist > 1e-6:
                        force = spring_k * (dist - 0.5) * weight * diff / dist
                        forces[i] += force
                        forces[j] -= force

            # Repulsion forces
            for i in range(len(node_list)):
                for j in range(i + 1, len(node_list)):
                    diff = node_list[i].position - node_list[j].position
                    dist = np.linalg.norm(diff)
                    if dist < 1.0 and dist > 1e-6:
                        force = -repulsion_k / (dist ** 2) * diff / dist
                        forces[i] += force
                        forces[j] -= force

            # Update positions
            for i, node in enumerate(node_list):
                pos = node.position + step_size * forces[i]
                norm = np.linalg.norm(pos)
                node.position = pos / (norm + 1e-8)

            if verbose and it % 20 == 0:
                stress = np.sum(np.abs(forces))
                print(f"  Iter {it}: stress={stress:.4f}")

        # Sync vectors back
        for node in node_list:
            node.vector = node.position.copy()

    # =========================================================================
    # v2.0: 子图管理
    # =========================================================================

    def __init__(self, dim=None, storage_dir=None):
        """初始化主图。

        Args:
            dim: 向量维度（auto-detected）
            storage_dir: 持久化目录，不指定则只存内存
        """
        self.dim = dim
        self.nodes = {}
        self.storage_dir = storage_dir
        self._subgraphs: dict[str, Subgraph] = {}   # 内存缓存
        self._projection_subgraphs: list[str] = []  # 投影空间中的子图ID列表
        self._projection_index: dict[str, dict] = {}  # 索引表

        if storage_dir:
            os.makedirs(os.path.join(storage_dir, "subgraphs"), exist_ok=True)
            self._load_subgraph_index()

    def _subgraph_path(self, sg_id: str) -> str:
        return os.path.join(self.storage_dir, "subgraphs", f"{sg_id}.json")

    def _load_subgraph_index(self):
        """从磁盘加载子图索引"""
        subgraphs_dir = os.path.join(self.storage_dir, "subgraphs")
        if not os.path.exists(subgraphs_dir):
            return
        for fname in os.listdir(subgraphs_dir):
            if fname.endswith(".json"):
                path = os.path.join(subgraphs_dir, fname)
                try:
                    d = json.loads(open(path).read())
                    sg = Subgraph.from_dict(d)
                    self._subgraphs[sg.id] = sg
                except Exception:
                    pass

    def _save_subgraph(self, sg: Subgraph):
        """持久化子图到磁盘"""
        if not self.storage_dir:
            return
        path = self._subgraph_path(sg.id)
        with open(path, "w") as f:
            json.dump(sg.to_dict(), f, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # 子图 CRUD
    # -------------------------------------------------------------------------

    def create_subgraph(self, task_id: str = "") -> str:
        """创建新子图，返回子图ID"""
        sg = Subgraph(task_id=task_id)
        self._subgraphs[sg.id] = sg
        self._save_subgraph(sg)
        return sg.id

    def get_subgraph(self, sg_id: str) -> Subgraph | None:
        """根据ID获取子图"""
        return self._subgraphs.get(sg_id)

    def list_subgraphs(self) -> list[Subgraph]:
        """列出所有子图"""
        return list(self._subgraphs.values())

    def set_subgraph_isolated(self, sg_id: str, isolated: bool) -> bool:
        """设置子图的隔离状态"""
        sg = self._subgraphs.get(sg_id)
        if not sg:
            return False
        sg.isolated = isolated
        sg.last_activated = time.time()
        self._save_subgraph(sg)
        return True

    def add_nodes_to_subgraph(self, sg_id: str, node_ids: list[str]) -> bool:
        """向子图添加节点"""
        sg = self._subgraphs.get(sg_id)
        if not sg:
            return False
        sg.add_nodes(node_ids)
        sg.last_activated = time.time()
        self._save_subgraph(sg)
        return True

    def add_edges_to_subgraph(self, sg_id: str, edges: list[tuple]) -> bool:
        """向子图添加边，edges 格式: [(src, tgt, weight), ...]"""
        sg = self._subgraphs.get(sg_id)
        if not sg:
            return False
        for e in edges:
            src, tgt, weight = e[0], e[1], e[2] if len(e) > 2 else 1.0
            sg.add_edge(src, tgt, weight)
        sg.last_activated = time.time()
        self._save_subgraph(sg)
        return True

    # -------------------------------------------------------------------------
    # 投影空间
    # -------------------------------------------------------------------------

    MAX_PROJECTION = 3  # 投影空间最多同时 3 个子图

    def load_into_projection(self, sg_id: str) -> bool:
        """将子图加载到投影空间（工作台）

        规则：
        - 已加载的子图直接返回 True（不做重复处理）
        - 容量满时，新子图进入 → evict 权重最低的已有子图（赢家是新的）
        - re-add 时，新子图权重 > 旧最低权重 → evict 旧最低，新 append
          新子图权重 <= 旧最低权重 → 新 append（超过容量触发下一次 evict）
        """
        if sg_id not in self._subgraphs:
            return False
        if sg_id in self._projection_subgraphs:
            # 已加载，直接更新激活时间
            sg = self._subgraphs[sg_id]
            sg.last_activated = time.time()
            return True

        new_sg = self._subgraphs[sg_id]
        new_weight = new_sg.weight

        # 如果容量已满，evict 权重最低的已有子图
        if len(self._projection_subgraphs) >= self.MAX_PROJECTION:
            min_weight = float("inf")
            min_id = None
            for sid in self._projection_subgraphs:
                sg = self._subgraphs.get(sid)
                if sg and sg.weight < min_weight:
                    min_weight = sg.weight
                    min_id = sid
            # 新子图权重 <= 旧最低：不 evict 新（而是 evict 最弱的已有）
            # 这个分支：新进来 evict 旧的最低，新 append
            if min_id:
                self._projection_subgraphs.remove(min_id)
                self._projection_index.pop(min_id, None)

        self._projection_subgraphs.append(sg_id)
        new_sg.last_activated = time.time()

        # 更新索引表
        self._projection_index[sg_id] = {
            "position": len(self._projection_subgraphs) - 1,
            "summary": f"任务: {new_sg.task_id}, 节点数: {len(new_sg.nodes)}",
            "node_count": len(new_sg.nodes),
            "weight": new_sg.weight,
        }

        self._save_subgraph(new_sg)
        self._save_projection()
        return True

    def get_projection_subgraphs(self) -> list[str]:
        """获取当前在投影空间中的子图ID列表"""
        # 验证子图仍然存在
        result = [sid for sid in self._projection_subgraphs if sid in self._subgraphs]
        if len(result) != len(self._projection_subgraphs):
            self._projection_subgraphs = result
        return list(result)

    def get_projection_index(self) -> dict[str, dict]:
        """获取投影空间索引表"""
        return dict(self._projection_index)

    def _save_projection(self):
        """保存投影空间状态到磁盘"""
        if not self.storage_dir:
            return
        path = os.path.join(self.storage_dir, "projection.json")
        data = {
            "subgraphs": self._projection_subgraphs,
            "index": self._projection_index,
        }
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False)

    def _load_projection(self):
        """从磁盘加载投影空间状态"""
        if not self.storage_dir:
            return
        path = os.path.join(self.storage_dir, "projection.json")
        if not os.path.exists(path):
            return
        try:
            data = json.loads(open(path).read())
            self._projection_subgraphs = data.get("subgraphs", [])
            self._projection_index = data.get("index", {})
        except Exception:
            pass

    def remove_from_projection(self, sg_id: str) -> bool:
        """手动将子图从投影空间移除"""
        if sg_id not in self._projection_subgraphs:
            return False
        self._projection_subgraphs.remove(sg_id)
        self._projection_index.pop(sg_id, None)
        self._save_projection()
        return True
