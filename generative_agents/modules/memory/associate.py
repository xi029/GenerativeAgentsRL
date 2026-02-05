"""generative_agents.memory.associate"""

import datetime
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever

from modules.storage.index import LlamaIndex
from modules import utils
from .event import Event


class Concept:
    def __init__(
        self,
        describe,
        node_id,
        node_type,
        subject,
        predicate,
        object,
        address,
        poignancy,
        create=None,
        expire=None,
        access=None,
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.event = Event(
            subject, predicate, object, describe=describe, address=address.split(":")
        )
        self.poignancy = poignancy
        self.create = utils.to_date(create) if create else utils.get_timer().get_date()
        if expire:
            self.expire = utils.to_date(expire)
        else:
            self.expire = self.create + datetime.timedelta(days=30)
        self.access = utils.to_date(access) if access else self.create

    def abstract(self):
        return {
            "{}(P.{})".format(self.node_type, self.poignancy): str(self.event),
            "duration": "{} ~ {} (access: {})".format(
                self.create.strftime("%Y%m%d-%H:%M"),
                self.expire.strftime("%Y%m%d-%H:%M"),
                self.access.strftime("%Y%m%d-%H:%M"),
            ),
        }

    def __str__(self):
        return utils.dump_dict(self.abstract())

    @property
    def describe(self):
        return self.event.get_describe()

    @classmethod
    def from_node(cls, node):
        return cls(node.text, node.id_, **node.metadata)

    @classmethod
    def from_event(cls, node_id, node_type, event, poignancy):
        return cls(
            event.get_describe(),
            node_id,
            node_type,
            event.subject,
            event.predicate,
            event.object,
            ":".join(event.address),
            poignancy,
        )


class AssociateRetriever(BaseRetriever):
    def __init__(self, config, task_query=None, *args, **kwargs) -> None:
        self._config = config
        self._task_query = task_query  # [NEW] 任务相关的查询词
        self._vector_retriever = VectorIndexRetriever(*args, **kwargs)
        super().__init__()

    def _retrieve(self, query_bundle):
        """Retrieve nodes given query."""

        # 1. 常规检索
        nodes = self._vector_retriever.retrieve(query_bundle)
        if not nodes:
            return []
            
        # [NEW] 2. 任务共鸣检索 (如果设置了任务)
        task_nodes_map = {}
        if self._task_query:
            # 使用任务描述额外检索一次，找出与任务高度相关的记忆
            # 为了效率，我们重用 vector_retriever 但不做过滤，或者简单做一次
            # 注意：这里实际上需要 embedding 匹配。
            # 简单起见，我们假设外部如果传入了 task_query，我们就在算分时给予 task_query 的相似度加成
            # 但 LlamaIndex 的 retrieve 已经返回了针对 query_bundle 的相似度 (node.score)
            
            # 更好的做法：我们直接在算分阶段，如果 node 的内容与 task_query 匹配，给予加分
            # 但这需要重新计算 embedding 相似度，比较慢。
            # 替代方案：检查 node 内容是否包含 task 关键词 (轻量级)
            pass

        nodes = sorted(
            nodes, key=lambda n: utils.to_date(n.metadata["access"]), reverse=True
        )
        # get scores
        fac = self._config["recency_decay"]
        recency_scores = self._normalize(
            [fac**i for i in range(1, len(nodes) + 1)], self._config["recency_weight"]
        )
        relevance_scores = self._normalize(
            [n.score for n in nodes], self._config["relevance_weight"]
        )
        importance_scores = self._normalize(
            [n.metadata["poignancy"] for n in nodes], self._config["importance_weight"]
        )
        
        # [NEW] Task Resonance Score
        task_scores = [0.0] * len(nodes)
        if self._task_query:
            # 简易版语义匹配：如果记忆包含任务关键词，加分
            # 比如任务是 "举办画展"，记忆里有 "画"、"展览" 则加分
            # 这里做一个简单的字符串匹配作为 "Lite Rerank"
            keywords = set(self._task_query.split())
            for i, node in enumerate(nodes):
                match_count = sum(1 for k in keywords if k in node.text)
                if match_count > 0:
                    task_scores[i] = 1.0  # 命中任务关键词，给予满额加成
        
        task_scores = self._normalize(
            task_scores, self._config.get("task_weight", 1.0) # 默认权重 1.0
        )

        final_scores = {
            n.id_: r1 + r2 + i + t
            for n, r1, r2, i, t in zip(
                nodes, recency_scores, relevance_scores, importance_scores, task_scores
            )
        }
        # re-rank nodes
        nodes = sorted(nodes, key=lambda n: final_scores[n.id_], reverse=True)
        nodes = nodes[: self._config["retrieve_max"]]
        for n in nodes:
            n.metadata["access"] = utils.get_timer().get_date("%Y%m%d-%H:%M:%S")
        return nodes

    def _normalize(self, data, factor=1, t_min=0, t_max=1):
        min_val, max_val = min(data), max(data)
        diff = max_val - min_val
        if diff == 0:
            return [(t_max - t_min) * factor / 2 for _ in data]
        return [(d - min_val) * (t_max - t_min) * factor / diff + t_min for d in data]


class Associate:
    def __init__(
        self,
        path,
        embedding,
        retention=8,
        max_memory=-1,
        max_importance=10,
        recency_decay=0.995,
        recency_weight=0.5,
        relevance_weight=3,
        importance_weight=2,
        memory=None,
    ):
        self._index = LlamaIndex(embedding, path)
        self.memory = memory or {"event": [], "thought": [], "chat": []}
        self.cleanup_index()
        self.retention = retention
        self.max_memory = max_memory
        self.max_importance = max_importance
        self._retrieve_config = {
            "recency_decay": recency_decay,
            "recency_weight": recency_weight,
            "relevance_weight": relevance_weight,
            "importance_weight": importance_weight,
        }

    def abstract(self):
        des = {"nodes": self._index.nodes_num}
        for t in ["event", "chat", "thought"]:
            des[t] = [self.find_concept(c).describe for c in self.memory[t]]
        return des

    def __str__(self):
        return utils.dump_dict(self.abstract())

    def cleanup_index(self):
        node_ids = self._index.cleanup()
        self.memory = {
            n_type: [n for n in nodes if n not in node_ids]
            for n_type, nodes in self.memory.items()
        }

    def add_node(
        self,
        node_type,
        event,
        poignancy,
        create=None,
        expire=None,
        filling=None,
    ):
        create = create or utils.get_timer().get_date()
        expire = expire or (create + datetime.timedelta(days=30))
        metadata = {
            "node_type": node_type,
            "subject": event.subject,
            "predicate": event.predicate,
            "object": event.object,
            "address": ":".join(event.address),
            "poignancy": poignancy,
            "create": create.strftime("%Y%m%d-%H:%M:%S"),
            "expire": expire.strftime("%Y%m%d-%H:%M:%S"),
            "access": create.strftime("%Y%m%d-%H:%M:%S"),
        }
        node = self._index.add_node(event.get_describe(), metadata)
        memory = self.memory[node_type]
        memory.insert(0, node.id_)
        if len(memory) >= self.max_memory > 0:
            self._index.remove_nodes(memory[self.max_memory:])
            self.memory[node_type] = memory[: self.max_memory - 1]
        return self.to_concept(node)

    def to_concept(self, node):
        return Concept.from_node(node)

    def find_concept(self, node_id):
        return self.to_concept(self._index.find_node(node_id))

    def _retrieve_nodes(self, node_type, text=None):
        if text:
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key="node_type", value=node_type)]
            )
            nodes = self._index.retrieve(
                text, filters=filters, node_ids=self.memory[node_type]
            )
        else:
            nodes = [self._index.find_node(n) for n in self.memory[node_type]]
        return [self.to_concept(n) for n in nodes[: self.retention]]

    def retrieve_events(self, text=None):
        return self._retrieve_nodes("event", text)

    def retrieve_thoughts(self, text=None):
        return self._retrieve_nodes("thought", text)

    def retrieve_chats(self, name=None):
        text = ("对话 " + name) if name else None
        return self._retrieve_nodes("chat", text)

    def retrieve_focus(self, focus, retrieve_max=30, reduce_all=True, task=None):
        def _create_retriever(*args, **kwargs):
            self._retrieve_config["retrieve_max"] = retrieve_max
            # [NEW] 传入任务上下文
            return AssociateRetriever(self._retrieve_config, task_query=task, *args, **kwargs)

        retrieved = {}
        node_ids = self.memory["event"] + self.memory["thought"]
        for text in focus:
            nodes = self._index.retrieve(
                text,
                similarity_top_k=len(node_ids),
                node_ids=node_ids,
                retriever_creator=_create_retriever,
            )
            if reduce_all:
                retrieved.update({n.id_: n for n in nodes})
            else:
                retrieved[text] = nodes
        if reduce_all:
            return [self.to_concept(v) for v in retrieved.values()]
        return {
            text: [self.to_concept(n) for n in nodes]
            for text, nodes, in retrieved.items()
        }

    def get_relation(self, node):
        return {
            "node": node,
            "events": self.retrieve_events(node.describe),
            "thoughts": self.retrieve_thoughts(node.describe),
        }

    def to_dict(self):
        self._index.save()
        return {"memory": self.memory}

    @property
    def index(self):
        return self._index
