"""
P0 Test: 单子图创建与存储

测试用例：
1. 子图创建 - 可以创建包含节点的子图
2. 子图存储 - 子图数据可以持久化到磁盘
3. 子图检索 - 可以从磁盘读取子图
4. 子图隔离 - 新创建的子图 isolated=True
5. 子图结构 - nodes/edges/task_id/weight/isolated 字段完整
"""
import os
import shutil
import tempfile
import pytest
import sys

sys.path.insert(0, "/home/linjunhe/.openclaw/workspace/projects/ldsg/src")

from master_graph import MasterSpatialGraph
from core_types import Node, Subgraph


class TestP0SubgraphCreation:
    """P0.1: 子图创建"""

    def test_create_empty_subgraph(self, tmp_path):
        """可以创建空子图"""
        sg = Subgraph(
            id="test-001",
            task_id="task-001",
            nodes=[],
            edges=[],
            weight=1.0,
            isolated=True,
        )
        assert sg.id == "test-001"
        assert sg.isolated is True
        assert sg.nodes == []

    def test_create_subgraph_with_nodes(self, tmp_path):
        """可以创建带节点的子图"""
        sg = Subgraph(
            id="test-002",
            task_id="task-002",
            nodes=["node-a", "node-b", "node-c"],
            edges=[("node-a", "node-b", 0.9)],
            weight=1.0,
            isolated=True,
        )
        assert len(sg.nodes) == 3
        assert len(sg.edges) == 1

    def test_subgraph_weight_defaults(self, tmp_path):
        """子图权重默认值为1.0"""
        sg = Subgraph(id="test-003", task_id="task-003")
        assert sg.weight == 1.0
        assert sg.isolated is True


class TestP0SubgraphStorage:
    """P0.2: 子图存储"""

    def test_save_and_load_subgraph(self, tmp_path):
        """子图可以持久化到磁盘并读取"""
        storage_dir = str(tmp_path / "ldsg")
        graph = MasterSpatialGraph(storage_dir=storage_dir)

        sg_id = graph.create_subgraph(task_id="task-storage-test")
        assert sg_id is not None

        # 读取子图
        sg = graph.get_subgraph(sg_id)
        assert sg is not None
        assert sg.id == sg_id
        assert sg.task_id == "task-storage-test"

    def test_save_multiple_subgraphs(self, tmp_path):
        """可以同时存储多个子图"""
        storage_dir = str(tmp_path / "ldsg2")
        graph = MasterSpatialGraph(storage_dir=storage_dir)

        sg1 = graph.create_subgraph(task_id="task-1")
        sg2 = graph.create_subgraph(task_id="task-2")
        sg3 = graph.create_subgraph(task_id="task-3")

        assert sg1 != sg2 != sg3

        # 全部可以读取
        assert graph.get_subgraph(sg1) is not None
        assert graph.get_subgraph(sg2) is not None
        assert graph.get_subgraph(sg3) is not None


class TestP0SubgraphIsolation:
    """P0.3: 子图隔离"""

    def test_new_subgraph_is_isolated(self, tmp_path):
        """新创建的子图默认是隔离的"""
        storage_dir = str(tmp_path / "ldsg3")
        graph = MasterSpatialGraph(storage_dir=storage_dir)

        sg_id = graph.create_subgraph(task_id="task-isolated")
        sg = graph.get_subgraph(sg_id)

        assert sg.isolated is True

    def test_can_toggle_isolation(self, tmp_path):
        """可以切换子图的隔离状态"""
        storage_dir = str(tmp_path / "ldsg4")
        graph = MasterSpatialGraph(storage_dir=storage_dir)

        sg_id = graph.create_subgraph(task_id="task-toggle")
        graph.set_subgraph_isolated(sg_id, False)
        sg = graph.get_subgraph(sg_id)
        assert sg.isolated is False

        graph.set_subgraph_isolated(sg_id, True)
        sg = graph.get_subgraph(sg_id)
        assert sg.isolated is True


class TestP0SubgraphStructure:
    """P0.4: 子图结构验证"""

    def test_subgraph_has_required_fields(self, tmp_path):
        """子图包含所有必要字段"""
        storage_dir = str(tmp_path / "ldsg5")
        graph = MasterSpatialGraph(storage_dir=storage_dir)

        sg_id = graph.create_subgraph(task_id="task-fields")
        sg = graph.get_subgraph(sg_id)

        # 必填字段
        assert hasattr(sg, "id")
        assert hasattr(sg, "task_id")
        assert hasattr(sg, "nodes")
        assert hasattr(sg, "edges")
        assert hasattr(sg, "weight")
        assert hasattr(sg, "isolated")
        assert hasattr(sg, "created_at")

    def test_subgraph_add_nodes(self, tmp_path):
        """可以向子图添加节点"""
        storage_dir = str(tmp_path / "ldsg6")
        graph = MasterSpatialGraph(storage_dir=storage_dir)

        sg_id = graph.create_subgraph(task_id="task-add-nodes")
        graph.add_nodes_to_subgraph(sg_id, ["node-x", "node-y"])

        sg = graph.get_subgraph(sg_id)
        assert "node-x" in sg.nodes
        assert "node-y" in sg.nodes

    def test_subgraph_add_edges(self, tmp_path):
        """可以向子图添加边"""
        storage_dir = str(tmp_path / "ldsg7")
        graph = MasterSpatialGraph(storage_dir=storage_dir)

        sg_id = graph.create_subgraph(task_id="task-add-edges")
        graph.add_nodes_to_subgraph(sg_id, ["node-a", "node-b"])
        graph.add_edges_to_subgraph(sg_id, [("node-a", "node-b", 0.8)])

        sg = graph.get_subgraph(sg_id)
        assert len(sg.edges) == 1


class TestP0ProjectionSpace:
    """P0.5: 子图加载到投影空间"""

    def test_load_subgraph_into_projection(self, tmp_path):
        """可以将子图加载到投影空间"""
        storage_dir = str(tmp_path / "ldsg8")
        graph = MasterSpatialGraph(storage_dir=storage_dir)

        sg_id = graph.create_subgraph(task_id="task-projection")
        graph.add_nodes_to_subgraph(sg_id, ["n1", "n2"])

        # 加载到投影空间
        ok = graph.load_into_projection(sg_id)
        assert ok is True
        assert sg_id in graph.get_projection_subgraphs()

    def test_projection_tracks_loaded_subgraphs(self, tmp_path):
        """投影空间能追踪已加载的子图"""
        storage_dir = str(tmp_path / "ldsg9")
        graph = MasterSpatialGraph(storage_dir=storage_dir)

        sg1 = graph.create_subgraph(task_id="p1")
        sg2 = graph.create_subgraph(task_id="p2")

        graph.load_into_projection(sg1)
        graph.load_into_projection(sg2)

        loaded = graph.get_projection_subgraphs()
        assert sg1 in loaded
        assert sg2 in loaded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
