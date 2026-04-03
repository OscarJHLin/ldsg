"""
P1 Test: 多子图投影空间管理

测试用例：
1. 投影空间最多 3 个子图
2. 第4个子图进入时，权重最低的自动挤出
3. 挤出不删除子图，只移出投影空间
4. 可以手动从投影空间移除子图
5. 投影空间满时，新子图进入会触发挤出
6. 索引表随投影空间同步更新
7. 权重更新后下次进入时按新权重排序
"""
import os
import pytest
import sys

sys.path.insert(0, "/home/linjunhe/.openclaw/workspace/projects/ldsg/src")
from master_graph import MasterSpatialGraph


class TestP1ProjectionCapacity:
    """P1.1: 投影空间容量限制"""

    def test_max_3_subgraphs_in_projection(self, tmp_path):
        """投影空间最多同时 3 个子图"""
        storage_dir = str(tmp_path / "ldsg")
        graph = MasterSpatialGraph(storage_dir=storage_dir)

        sg1 = graph.create_subgraph(task_id="t1")
        sg2 = graph.create_subgraph(task_id="t2")
        sg3 = graph.create_subgraph(task_id="t3")
        sg4 = graph.create_subgraph(task_id="t4")

        graph.load_into_projection(sg1)
        graph.load_into_projection(sg2)
        graph.load_into_projection(sg3)
        graph.load_into_projection(sg4)  # 触发挤出

        loaded = graph.get_projection_subgraphs()
        assert len(loaded) <= 3

    def test_lru_eviction_when_exceed_3(self, tmp_path):
        """第4个进入时，权重最低的挤出"""
        storage_dir = str(tmp_path / "ldsg")
        graph = MasterSpatialGraph(storage_dir=storage_dir)

        sg1 = graph.create_subgraph(task_id="t1")
        sg2 = graph.create_subgraph(task_id="t2")
        sg3 = graph.create_subgraph(task_id="t3")
        sg4 = graph.create_subgraph(task_id="t4")

        # sg3 weight=0.5 (最低), sg1=1.0, sg2=0.8
        g3 = graph.get_subgraph(sg3)
        g3.weight = 0.5

        graph.load_into_projection(sg1)  # projection: [sg1]
        graph.load_into_projection(sg2)  # projection: [sg1, sg2]
        graph.load_into_projection(sg3)  # projection: [sg1, sg2, sg3]
        # 第4个进入时，sg3(0.5) 最低被挤出
        ok = graph.load_into_projection(sg4)  # projection: [sg1, sg2, sg4]

        loaded = graph.get_projection_subgraphs()
        assert sg3 not in loaded  # sg3 被挤出
        assert sg4 in loaded      # sg4 在里面
        assert sg1 in loaded     # sg1 保留
        assert sg2 in loaded     # sg2 保留

    def test_eviction_does_not_delete_subgraph(self, tmp_path):
        """挤出只移出投影空间，不删除子图"""
        storage_dir = str(tmp_path / "ldsg")
        graph = MasterSpatialGraph(storage_dir=storage_dir)

        sg1 = graph.create_subgraph(task_id="t1")
        sg2 = graph.create_subgraph(task_id="t2")
        sg3 = graph.create_subgraph(task_id="t3")
        sg4 = graph.create_subgraph(task_id="t4")

        g3 = graph.get_subgraph(sg3)
        g3.weight = 0.5  # sg3 权重最低，会被挤出

        graph.load_into_projection(sg1)
        graph.load_into_projection(sg2)
        graph.load_into_projection(sg3)
        graph.load_into_projection(sg4)  # 挤出 sg3

        # sg3 还在，只是移出投影空间
        sg = graph.get_subgraph(sg3)
        assert sg is not None  # 子图仍然存在


class TestP1ProjectionIndex:
    """P1.2: 投影空间索引表"""

    def test_index_updated_on_load(self, tmp_path):
        """加载子图到投影空间时，索引表同步更新"""
        storage_dir = str(tmp_path / "ldsg")
        graph = MasterSpatialGraph(storage_dir=storage_dir)

        sg_id = graph.create_subgraph(task_id="test-index")
        graph.add_nodes_to_subgraph(sg_id, ["n1", "n2"])
        graph.load_into_projection(sg_id)

        index = graph.get_projection_index()
        assert sg_id in index
        assert index[sg_id]["node_count"] == 2
        assert "summary" in index[sg_id]

    def test_index_updated_on_eviction(self, tmp_path):
        """挤出时索引表同步删除条目"""
        storage_dir = str(tmp_path / "ldsg")
        graph = MasterSpatialGraph(storage_dir=storage_dir)

        sg1 = graph.create_subgraph(task_id="t1")
        sg2 = graph.create_subgraph(task_id="t2")
        sg3 = graph.create_subgraph(task_id="t3")
        sg4 = graph.create_subgraph(task_id="t4")

        g3 = graph.get_subgraph(sg3)
        g3.weight = 0.5  # sg3 最低

        graph.load_into_projection(sg1)
        graph.load_into_projection(sg2)
        graph.load_into_projection(sg3)
        graph.load_into_projection(sg4)  # 挤出 sg3

        index = graph.get_projection_index()
        assert sg1 in index
        assert sg2 in index
        assert sg3 not in index  # 被挤出，索引也删除


class TestP1ManualRemove:
    """P1.3: 手动移除"""

    def test_remove_from_projection(self, tmp_path):
        """可以手动将子图从投影空间移除"""
        storage_dir = str(tmp_path / "ldsg")
        graph = MasterSpatialGraph(storage_dir=storage_dir)

        sg1 = graph.create_subgraph(task_id="t1")
        sg2 = graph.create_subgraph(task_id="t2")

        graph.load_into_projection(sg1)
        graph.load_into_projection(sg2)

        ok = graph.remove_from_projection(sg1)
        assert ok is True
        assert sg1 not in graph.get_projection_subgraphs()
        assert sg2 in graph.get_projection_subgraphs()

    def test_remove_nonexistent_returns_false(self, tmp_path):
        """移除不存在的子图返回 False"""
        storage_dir = str(tmp_path / "ldsg")
        graph = MasterSpatialGraph(storage_dir=storage_dir)
        ok = graph.remove_from_projection("nonexistent-id")
        assert ok is False


class TestP1WeightUpdate:
    """P1.4: 权重更新"""

    def test_weight_change_affects_eviction_order(self, tmp_path):
        """权重更新后，下次进入时按新权重决定谁被挤出"""
        storage_dir = str(tmp_path / "ldsg")
        graph = MasterSpatialGraph(storage_dir=storage_dir)

        sg1 = graph.create_subgraph(task_id="t1")
        sg2 = graph.create_subgraph(task_id="t2")
        sg3 = graph.create_subgraph(task_id="t3")
        sg4 = graph.create_subgraph(task_id="t4")

        g1 = graph.get_subgraph(sg1); g1.weight = 1.0
        g2 = graph.get_subgraph(sg2); g2.weight = 0.8
        g3 = graph.get_subgraph(sg3); g3.weight = 0.5  # 最低

        graph.load_into_projection(sg1)  # [sg1]
        graph.load_into_projection(sg2)  # [sg1, sg2]
        graph.load_into_projection(sg3)  # [sg1, sg2, sg3]; sg3 最低

        # sg4 进入，sg3(0.5) 被挤出
        ok = graph.load_into_projection(sg4)
        loaded = graph.get_projection_subgraphs()

        assert sg4 in loaded
        assert sg3 not in loaded  # sg3 最低被挤出
        assert sg1 in loaded
        assert sg2 in loaded


class TestP1ReEnter:
    """P1.5: 重复进入"""

    def test_already_loaded_returns_true(self, tmp_path):
        """已在投影空间的子图重复 load 不报错"""
        storage_dir = str(tmp_path / "ldsg")
        graph = MasterSpatialGraph(storage_dir=storage_dir)

        sg_id = graph.create_subgraph(task_id="t1")
        ok1 = graph.load_into_projection(sg_id)
        ok2 = graph.load_into_projection(sg_id)

        assert ok1 is True
        assert ok2 is True
        assert len(graph.get_projection_subgraphs()) == 1

    def test_readd_after_eviction(self, tmp_path):
        """被挤出的子图可以重新进入"""
        storage_dir = str(tmp_path / "ldsg")
        graph = MasterSpatialGraph(storage_dir=storage_dir)

        sg1 = graph.create_subgraph(task_id="t1")  # weight=1.0
        sg2 = graph.create_subgraph(task_id="t2")  # weight=1.0
        sg3 = graph.create_subgraph(task_id="t3")  # weight=1.0 default
        sg4 = graph.create_subgraph(task_id="t4")  # weight=1.0 default

        # 先加载 3 个
        graph.load_into_projection(sg1)  # [sg1]
        graph.load_into_projection(sg2)  # [sg1, sg2]
        graph.load_into_projection(sg3)  # [sg1, sg2, sg3]

        # sg4 进入，sg1/sg2/sg3 权重都是 1.0，第一个（sg1）被挤出
        graph.load_into_projection(sg4)
        assert sg1 not in graph.get_projection_subgraphs()

        # 更新 sg1 权重为 0.95，重新加载
        # sg1 进入满的投影，sg2(weight 1.0, 第一个)被挤出
        # 结果：[sg3, sg4, sg1]
        g1 = graph.get_subgraph(sg1)
        g1.weight = 0.95
        graph.load_into_projection(sg1)

        assert sg1 in graph.get_projection_subgraphs()  # sg1 在
        assert sg3 in graph.get_projection_subgraphs()  # sg3 也在
        assert sg2 not in graph.get_projection_subgraphs()  # sg2 被挤出


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
