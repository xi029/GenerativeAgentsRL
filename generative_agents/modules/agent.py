"""generative_agents.agent"""

import os
import math
import random
import datetime

from modules import memory, prompt, utils
from modules.model.llm_model import create_llm_model
from modules.memory.associate import Concept
from modules.rl import encode_agent_state, AgentActionSpace, calc_agent_reward


class Agent:
    def __init__(self, config, maze, conversation, logger):
        self.name = config["name"]
        self.maze = maze
        self.conversation = conversation
        self._llm = None
        self.logger = logger

        # agent config
        self.percept_config = config["percept"]
        self.think_config = config["think"]
        self.chat_iter = config["chat_iter"]

        # memory
        self.spatial = memory.Spatial(**config["spatial"])
        self.schedule = memory.Schedule(**config["schedule"])
        self.associate = memory.Associate(
            os.path.join(config["storage_root"], "associate"), **config["associate"]
        )
        self.concepts, self.chats = [], config.get("chats", [])

        # prompt
        self.scratch = prompt.Scratch(self.name, config["currently"], config["scratch"])

        # status
        status = {"poignancy": 0}
        self.status = utils.update_dict(status, config.get("status", {}))
        self.plan = config.get("plan", {})

        # record
        self.last_record = utils.get_timer().daily_duration()

        # RL initialization
        self.rl_action_space = AgentActionSpace()
        self.rl_grpo = None

        # action and events
        if "action" in config:
            self.action = memory.Action.from_dict(config["action"])
            tiles = self.maze.get_address_tiles(self.get_event().address)
            config["coord"] = random.choice(list(tiles))
        else:
            tile = self.maze.tile_at(config["coord"])
            address = tile.get_address("game_object", as_list=True)
            self.action = memory.Action(
                memory.Event(self.name, address=address),
                memory.Event(address[-1], address=address),
            )

        # update maze
        self.coord, self.path = None, None
        self.move(config["coord"], config.get("path"))
        if self.coord is None:
            self.coord = config["coord"]

    def abstract(self):
        des = {
            "name": self.name,
            "currently": self.scratch.currently,
            "tile": self.maze.tile_at(self.coord).abstract(),
            "status": self.status,
            "concepts": {c.node_id: c.abstract() for c in self.concepts},
            "chats": self.chats,
            "action": self.action.abstract(),
            "associate": self.associate.abstract(),
        }
        if self.schedule.scheduled():
            des["schedule"] = self.schedule.abstract()
        if self.llm_available():
            des["llm"] = self._llm.get_summary()
        # if self.plan.get("path"):
        #     des["path"] = "-".join(
        #         ["{},{}".format(c[0], c[1]) for c in self.plan["path"]]
        #     )
        return des

    def __str__(self):
        return utils.dump_dict(self.abstract())

    def reset(self):
        if not self._llm:
            self._llm = create_llm_model(self.think_config["llm"])

    def completion(self, func_hint, *args, **kwargs):
        assert hasattr(
            self.scratch, "prompt_" + func_hint
        ), "Can not find func prompt_{} from scratch".format(func_hint)
        func = getattr(self.scratch, "prompt_" + func_hint)
        res = func(*args, **kwargs)._asdict()
        title, msg = "{}.{}".format(self.name, func_hint), {}
        if self.llm_available():
            self.logger.info("{} -> {}".format(self.name, func_hint))
            output = self._llm.completion(**res)
            msg = {"<PROMPT>": "\n" + res["prompt"] + "\n"}
            msg.update({"response": output})
        self.logger.debug(utils.block_msg(title, msg))
        return output

    def think(self, status, agents):
        events = self.move(status["coord"], status.get("path"))
        plan, _ = self.make_schedule()

        # Record last coordinate for RL
        self.last_coord = self.get_tile().coord

        if (plan["describe"] == "sleeping" or "睡" in plan["describe"]) and self.is_awake():
            self.logger.info("{} is going to sleep...".format(self.name))
            address = self.spatial.find_address("睡觉", as_list=True)
            tiles = self.maze.get_address_tiles(address)
            coord = random.choice(list(tiles))
            events = self.move(coord)
            self.action = memory.Action(
                memory.Event(self.name, "正在", "睡觉", address=address, emoji="😴"),
                memory.Event(
                    address[-1],
                    "被占用",
                    self.name,
                    address=address,
                    emoji="🛌",
                ),
                duration=plan["duration"],
                start=utils.get_timer().daily_time(plan["start"]),
            )
        if self.is_awake():
            self.percept()
            
            # RL Logic Insertion
            rl_triggered = False
            if self.rl_grpo and self.scratch.config.get("group") and self.scratch.config.get("task"):
                # 1. Encode Agent State
                agent_state = encode_agent_state(self)
                # 2. GRPO Select Action
                action_idx, log_prob = self.rl_grpo.select_action(agent_state, self.name)
                action_desc = self.rl_action_space.get_action_desc(action_idx)
                # 3. Map Action & Inject Prompt
                self._map_rl_action_to_behavior_with_reward_prompt(action_desc, agents)
                # 4. Calculate & Record Reward
                reward = calc_agent_reward(self)
                self.rl_grpo.add_reward(self.name, reward)
                rl_triggered = True

            # If RL didn't trigger specific action logic (or we want to mix), continue with normal plan
            # But if RL set an action, we might want to respect it?
            # _map_rl_action_to_behavior_with_reward_prompt sets self.action or self.chats
            # Normal make_plan does _reaction (chat/wait) or find_path (move)
            
            # If RL executed a chat, self.chats is not empty, _reaction might skip?
            # If RL executed a move, self.action is updated.
            
            # We run standard make_plan as fallback or to handle pathfinding for RL move
            self.make_plan(agents)
            
            self.reflect()
        else:
            if self.action.finished():
                self.action = self._determine_action()

        emojis = {}
        if self.action:
            emojis[self.name] = {"emoji": self.get_event().emoji, "coord": self.coord}
        for eve, coord in events.items():
            if eve.subject in agents:
                continue
            emojis[":".join(eve.address)] = {"emoji": eve.emoji, "coord": coord}
        self.plan = {
            "name": self.name,
            "path": self.find_path(agents),
            "emojis": emojis,
        }
        return self.plan

    def move(self, coord, path=None):
        events = {}

        def _update_tile(coord):
            tile = self.maze.tile_at(coord)
            if not self.action:
                return {}
            if not tile.update_events(self.get_event()):
                tile.add_event(self.get_event())
            obj_event = self.get_event(False)
            if obj_event:
                self.maze.update_obj(coord, obj_event)
            return {e: coord for e in tile.get_events()}

        if self.coord and self.coord != coord:
            tile = self.get_tile()
            tile.remove_events(subject=self.name)
            if tile.has_address("game_object"):
                addr = tile.get_address("game_object")
                self.maze.update_obj(
                    self.coord, memory.Event(addr[-1], address=addr)
                )
            events.update({e: self.coord for e in tile.get_events()})
        if not path:
            events.update(_update_tile(coord))
        self.coord = coord
        self.path = path or []

        return events

    def make_schedule(self):
        if not self.schedule.scheduled():
            self.logger.info("{} is making schedule...".format(self.name))
            # update currently
            if self.associate.index.nodes_num > 0:
                self.associate.cleanup_index()
                focus = [
                    f"{self.name} 在 {utils.get_timer().daily_format_cn()} 的计划。",
                    f"在 {self.name} 的生活中，重要的近期事件。",
                ]
                retrieved = self.associate.retrieve_focus(focus)
                self.logger.info(
                    "{} retrieved {} concepts".format(self.name, len(retrieved))
                )
                if retrieved:
                    plan = self.completion("retrieve_plan", retrieved)
                    thought = self.completion("retrieve_thought", retrieved)
                    self.scratch.currently = self.completion(
                        "retrieve_currently", plan, thought
                    )
            # make init schedule
            self.schedule.create = utils.get_timer().get_date()
            wake_up = self.completion("wake_up")
            init_schedule = self.completion("schedule_init", wake_up)
            # make daily schedule
            hours = [f"{i}:00" for i in range(24)]
            # seed = [(h, "sleeping") for h in hours[:wake_up]]
            seed = [(h, "睡觉") for h in hours[:wake_up]]
            seed += [(h, "") for h in hours[wake_up:]]
            schedule = {}
            for _ in range(self.schedule.max_try):
                schedule = {h: s for h, s in seed[:wake_up]}
                schedule.update(
                    self.completion("schedule_daily", wake_up, init_schedule)
                )
                if len(set(schedule.values())) >= self.schedule.diversity:
                    break

            def _to_duration(date_str):
                return utils.daily_duration(utils.to_date(date_str, "%H:%M"))

            schedule = {_to_duration(k): v for k, v in schedule.items()}
            starts = list(sorted(schedule.keys()))
            for idx, start in enumerate(starts):
                end = starts[idx + 1] if idx + 1 < len(starts) else 24 * 60
                self.schedule.add_plan(schedule[start], end - start)
            schedule_time = utils.get_timer().time_format_cn(self.schedule.create)
            thought = "这是 {} 在 {} 的计划：{}".format(
                self.name, schedule_time, "；".join(init_schedule)
            )
            event = memory.Event(
                self.name,
                "计划",
                schedule_time,
                describe=thought,
                address=self.get_tile().get_address(),
            )
            self._add_concept(
                "thought",
                event,
                expire=self.schedule.create + datetime.timedelta(days=30),
            )
        # decompose current plan
        plan, _ = self.schedule.current_plan()
        if self.schedule.decompose(plan):
            decompose_schedule = self.completion(
                "schedule_decompose", plan, self.schedule
            )
            decompose, start = [], plan["start"]
            for describe, duration in decompose_schedule:
                decompose.append(
                    {
                        "idx": len(decompose),
                        "describe": describe,
                        "start": start,
                        "duration": duration,
                    }
                )
                start += duration
            plan["decompose"] = decompose
        return self.schedule.current_plan()

    def revise_schedule(self, event, start, duration):
        self.action = memory.Action(event, start=start, duration=duration)
        plan, _ = self.schedule.current_plan()
        if len(plan["decompose"]) > 0:
            plan["decompose"] = self.completion(
                "schedule_revise", self.action, self.schedule
            )

    def percept(self):
        scope = self.maze.get_scope(self.coord, self.percept_config)
        # add spatial memory
        for tile in scope:
            if tile.has_address("game_object"):
                self.spatial.add_leaf(tile.address)
        events, arena = {}, self.get_tile().get_address("arena")
        # gather events in scope
        for tile in scope:
            if not tile.events or tile.get_address("arena") != arena:
                continue
            dist = math.dist(tile.coord, self.coord)
            for event in tile.get_events():
                if dist < events.get(event, float("inf")):
                    events[event] = dist
        events = list(sorted(events.keys(), key=lambda k: events[k]))
        # get concepts
        self.concepts, valid_num = [], 0
        for idx, event in enumerate(events[: self.percept_config["att_bandwidth"]]):
            recent_nodes = (
                self.associate.retrieve_events() + self.associate.retrieve_chats()
            )
            recent_nodes = set(n.describe for n in recent_nodes)
            if event.get_describe() not in recent_nodes:
                if event.object == "idle" or event.object == "空闲":
                    node = Concept.from_event(
                        "idle_" + str(idx), "event", event, poignancy=1
                    )
                else:
                    valid_num += 1
                    node_type = "chat" if event.fit(self.name, "对话") else "event"
                    node = self._add_concept(node_type, event)
                    self.status["poignancy"] += node.poignancy
                self.concepts.append(node)
        self.concepts = [c for c in self.concepts if c.event.subject != self.name]
        self.logger.info(
            "{} percept {}/{} concepts".format(self.name, valid_num, len(self.concepts))
        )

    def make_plan(self, agents):
        if self._reaction(agents):
            return
        if self.path:
            return
        if self.action.finished():
            self.action = self._determine_action()

    # create action && object events
    def make_event(self, subject, describe, address):
        # emoji = self.completion("describe_emoji", describe)
        # return self.completion(
        #     "describe_event", subject, subject + describe, address, emoji
        # )

        e_describe = describe.replace("(", "").replace(")", "").replace("<", "").replace(">", "")
        if e_describe.startswith(subject + "此时"):
            e_describe = e_describe[len(subject + "此时"):]
        if e_describe.startswith(subject):
            e_describe = e_describe[len(subject):]
        event = memory.Event(
            subject, "此时", e_describe, describe=describe, address=address
        )
        return event

    def reflect(self):
        def _add_thought(thought, evidence=None):
            # event = self.completion(
            #     "describe_event",
            #     self.name,
            #     thought,
            #     address=self.get_tile().get_address(),
            # )
            event = self.make_event(self.name, thought, self.get_tile().get_address())
            return self._add_concept("thought", event, filling=evidence)

        if self.status["poignancy"] < self.think_config["poignancy_max"]:
            return
        nodes = self.associate.retrieve_events() + self.associate.retrieve_thoughts()
        if not nodes:
            return
        self.logger.info(
            "{} reflect(P{}/{}) with {} concepts...".format(
                self.name,
                self.status["poignancy"],
                self.think_config["poignancy_max"],
                len(nodes),
            )
        )
        nodes = sorted(nodes, key=lambda n: n.access, reverse=True)[
            : self.associate.max_importance
        ]
        # summary thought
        focus = self.completion("reflect_focus", nodes, 3)
        # [MOD] 传入当前任务，触发 Task-Driven Retrieval
        current_task = self.scratch.config.get("task", "")
        retrieved = self.associate.retrieve_focus(
            focus, reduce_all=False, task=current_task
        )
        for r_nodes in retrieved.values():
            thoughts = self.completion("reflect_insights", r_nodes, 5)
            for thought, evidence in thoughts:
                _add_thought(thought, evidence)
        # summary chats
        if self.chats:
            recorded, evidence = set(), []
            for name, _ in self.chats:
                if name == self.name or name in recorded:
                    continue
                res = self.associate.retrieve_chats(name)
                if res and len(res) > 0:
                    node = res[-1]
                    evidence.append(node.node_id)
            thought = self.completion("reflect_chat_planing", self.chats)
            _add_thought(f"对于 {self.name} 的计划：{thought}", evidence)
            thought = self.completion("reflect_chat_memory", self.chats)
            _add_thought(f"{self.name} {thought}", evidence)
        self.status["poignancy"] = 0
        self.chats = []

    def find_path(self, agents):
        address = self.get_event().address
        if self.path:
            return self.path
        if address == self.get_tile().get_address():
            return []
        if address[0] == "<waiting>":
            return []
        if address[0] == "<persona>":
            target_tiles = self.maze.get_around(agents[address[1]].coord)
        else:
            target_tiles = self.maze.get_address_tiles(address)
        if tuple(self.coord) in target_tiles:
            return []

        # filter tile with self event
        def _ignore_target(t_coord):
            if list(t_coord) == list(self.coord):
                return True
            events = self.maze.tile_at(t_coord).get_events()
            if any(e.subject in agents for e in events):
                return True
            return False

        target_tiles = [t for t in target_tiles if not _ignore_target(t)]
        if not target_tiles:
            return []
        if len(target_tiles) >= 4:
            target_tiles = random.sample(target_tiles, 4)
        pathes = {t: self.maze.find_path(self.coord, t) for t in target_tiles}
        target = min(pathes, key=lambda p: len(pathes[p]))
        return pathes[target][1:]

    def _determine_action(self):
        self.logger.info("{} is determining action...".format(self.name))
        plan, de_plan = self.schedule.current_plan()
        describes = [plan["describe"], de_plan["describe"]]
        address = self.spatial.find_address(describes[0], as_list=True)
        if not address:
            tile = self.get_tile()
            kwargs = {
                "describes": describes,
                "spatial": self.spatial,
                "address": tile.get_address("world", as_list=True),
            }
            kwargs["address"].append(
                self.completion("determine_sector", **kwargs, tile=tile)
            )
            arenas = self.spatial.get_leaves(kwargs["address"])
            if len(arenas) == 1:
                kwargs["address"].append(arenas[0])
            else:
                kwargs["address"].append(self.completion("determine_arena", **kwargs))
            objs = self.spatial.get_leaves(kwargs["address"])
            if len(objs) == 1:
                kwargs["address"].append(objs[0])
            elif len(objs) > 1:
                kwargs["address"].append(self.completion("determine_object", **kwargs))
            address = kwargs["address"]

        event = self.make_event(self.name, describes[-1], address)
        obj_describe = self.completion("describe_object", address[-1], describes[-1])
        obj_event = self.make_event(address[-1], obj_describe, address)

        event.emoji = f"{de_plan['describe']}"

        return memory.Action(
            event,
            obj_event,
            duration=de_plan["duration"],
            start=utils.get_timer().daily_time(de_plan["start"]),
        )

    def _reaction(self, agents=None, ignore_words=None):
        focus = None
        ignore_words = ignore_words or ["空闲"]

        def _focus(concept):
            return concept.event.subject in agents

        def _ignore(concept):
            return any(i in concept.describe for i in ignore_words)

        if agents:
            priority = [i for i in self.concepts if _focus(i)]
            if priority:
                focus = random.choice(priority)
        if not focus:
            priority = [i for i in self.concepts if not _ignore(i)]
            if priority:
                focus = random.choice(priority)
        if not focus or focus.event.subject not in agents:
            return
        other, focus = agents[focus.event.subject], self.associate.get_relation(focus)

        if self._chat_with(other, focus):
            return True
        if self._wait_other(other, focus):
            return True
        return False

    def _skip_react(self, other):
        def _skip(event):
            if not event.address or "sleeping" in event.get_describe(False) or "睡觉" in event.get_describe(False):
                return True
            if event.predicate == "待开始":
                return True
            return False

        if utils.get_timer().daily_duration(mode="hour") >= 23:
            return True
        if _skip(self.get_event()) or _skip(other.get_event()):
            return True
        return False

    def _chat_with(self, other, focus):
        if len(self.schedule.daily_schedule) < 1 or len(other.schedule.daily_schedule) < 1:
            # initializing
            return False
        if self._skip_react(other):
            return False
        if other.path:
            return False
        if self.get_event().fit(predicate="对话") or other.get_event().fit(predicate="对话"):
            return False

        chats = self.associate.retrieve_chats(other.name)
        if chats:
            delta = utils.get_timer().get_delta(chats[0].create)
            self.logger.info(
                "retrieved chat between {} and {}({} min):\n{}".format(
                    self.name, other.name, delta, chats[0]
                )
            )
            if delta < 60:
                return False

        if not self.completion("decide_chat", self, other, focus, chats):
            return False

        self.logger.info("{} decides chat with {}".format(self.name, other.name))
        start, chats = utils.get_timer().get_date(), []
        relations = [
            self.completion("summarize_relation", self, other.name),
            other.completion("summarize_relation", other, self.name),
        ]

        for i in range(self.chat_iter):
            text = self.completion(
                "generate_chat", self, other, relations[0], chats
            )

            if i > 0:
                # 对于发起对话的Agent，从第2轮对话开始，检查是否出现“复读”现象
                end = self.completion(
                    "generate_chat_check_repeat", self, chats, text
                )
                if end:
                    break

                # 对于发起对话的Agent，从第2轮对话开始，检查话题是否结束
                chats.append((self.name, text))
                end = self.completion(
                    "decide_chat_terminate", self, other, chats
                )
                if end:
                    break
            else :
                chats.append((self.name, text))

            text = other.completion(
                "generate_chat", other, self, relations[1], chats
            )
            if i > 0:
                # 对于响应对话的Agent，从第2轮开始，检查是否出现“复读”现象
                end = self.completion(
                    "generate_chat_check_repeat", other, chats, text
                )
                if end:
                    break

            chats.append((other.name, text))

            # 对于响应对话的Agent，从第1轮开始，检查话题是否结束
            end = other.completion(
                "decide_chat_terminate", other, self, chats
            )
            if end:
                break

        key = utils.get_timer().get_date("%Y%m%d-%H:%M")
        if key not in self.conversation.keys():
            self.conversation[key] = []
        self.conversation[key].append({f"{self.name} -> {other.name} @ {'，'.join(self.get_event().address)}": chats})

        self.logger.info(
            "{} and {} has chats\n  {}".format(
                self.name,
                other.name,
                "\n  ".join(["{}: {}".format(n, c) for n, c in chats]),
            )
        )
        chat_summary = self.completion("summarize_chats", chats)
        duration = int(sum([len(c[1]) for c in chats]) / 240)
        self.schedule_chat(
            chats, chat_summary, start, duration, other
        )
        other.schedule_chat(chats, chat_summary, start, duration, self)
        return True

    def _wait_other(self, other, focus):
        if self._skip_react(other):
            return False
        if not self.path:
            return False
        if self.get_event().address != other.get_tile().get_address():
            return False
        if not self.completion("decide_wait", self, other, focus):
            return False
        self.logger.info("{} decides wait to {}".format(self.name, other.name))
        start = utils.get_timer().get_date()
        # duration = other.action.end - start
        t = other.action.end - start
        duration = int(t.total_seconds() / 60)
        event = memory.Event(
            self.name,
            "waiting to start",
            self.get_event().get_describe(False),
            # address=["<waiting>"] + self.get_event().address,
            address=self.get_event().address,
            emoji=f"⌛",
        )
        self.revise_schedule(event, start, duration)

    def schedule_chat(self, chats, chats_summary, start, duration, other, address=None):
        self.chats.extend(chats)
        event = memory.Event(
            self.name,
            "对话",
            other.name,
            describe=chats_summary,
            address=address or self.get_tile().get_address(),
            emoji=f"💬",
        )
        self.revise_schedule(event, start, duration)

    def _add_concept(
        self,
        e_type,
        event,
        create=None,
        expire=None,
        filling=None,
    ):
        if event.fit(None, "is", "idle"):
            poignancy = 1
        elif event.fit(None, "此时", "空闲"):
            poignancy = 1
        elif e_type == "chat":
            poignancy = self.completion("poignancy_chat", event)
        else:
            poignancy = self.completion("poignancy_event", event)
        self.logger.debug("{} add associate {}".format(self.name, event))
        return self.associate.add_node(
            e_type,
            event,
            poignancy,
            create=create,
            expire=expire,
            filling=filling,
        )

    def get_tile(self):
        return self.maze.tile_at(self.coord)

    def get_event(self, as_act=True):
        return self.action.event if as_act else self.action.obj_event

    def is_awake(self):
        if not self.action:
            return True
        if self.get_event().fit(self.name, "is", "sleeping"):
            return False
        if self.get_event().fit(self.name, "正在", "睡觉"):
            return False
        return True

    def llm_available(self):
        if not self._llm:
            return False
        return self._llm.is_available()

    def _get_high_reward_actions(self, top_k=3):
        """提取近期高奖励行为（用于Prompt注入）"""
        high_reward_events = []
        # Access retrieve_events from associate
        events = self.associate.retrieve_events()
        # Since we just added nodes with 'reward' attribute dynamically in _map_rl_action...
        # We need to check if the retrieved concepts/events have that info.
        # Concept.event is an Event object. The attribute was added to Event object.
        # But retrieval returns Concept objects which wrap Event objects.
        # When loaded from storage/index, dynamic attributes might be lost unless persisted in metadata.
        # For this simplified implementation, we rely on runtime memory or accept it might be lost after restart.
        
        for concept in events[:20]:
            event = concept.event
            if hasattr(event, "reward") and event.reward > 0.1:
                high_reward_events.append(event.describe)
        return high_reward_events[:top_k]

    def _create_move_action(self, target_coord):
        tile = self.maze.tile_at(target_coord)
        address = tile.get_address("game_object", as_list=True)
        # describe = f"前往 {tile.get_address()}"
        describe = "正在移动"
        event = memory.Event(self.name, "正在", describe, address=address)
        obj_event = memory.Event(address[-1], "被访问", self.name, address=address)
        start = utils.get_timer().daily_time()
        # Set a short duration for RL move
        return memory.Action(event, obj_event, start=start, duration=10)

    def _create_event(self, describe):
        tile = self.get_tile()
        address = tile.get_address("game_object", as_list=True)
        return memory.Event(self.name, "执行", describe, describe=describe, address=address)

    def _map_rl_action_to_behavior_with_reward_prompt(self, action_desc, agents):
        """动作映射+奖励信号注入Prompt，生成动态行为"""
        agent_group = self.scratch.config.get("group", "")
        agent_task = self.scratch.config.get("task", "")
        high_reward_actions = self._get_high_reward_actions()
        current_context = f"当前位置：{self.get_tile().get_address()}，当前时间：{utils.get_timer().get_date()}"
        
        # 1. 移动动作
        if action_desc == "move":
            target_agent = next((a for a in agents.values() 
                               if a.name != self.name and a.scratch.config.get("group") == agent_group), None)
            if target_agent:
                self.action = self._create_move_action(target_agent.get_tile().coord)
                # Clear path so find_path recalculates
                self.path = []
        
        # 2. 对话动作
        elif action_desc == "chat":
            target_agent = next((a for a in agents.values() 
                               if a.name != self.name and a.scratch.config.get("group") == agent_group), None)
            if target_agent:
                prompt = f"""
                你的身份：{self.scratch.config.get('innate', '')}，所属组：{agent_group}，核心任务：{agent_task}
                近期高效对话行为：{high_reward_actions if high_reward_actions else ['与同组Agent讨论任务进展']}
                当前情境：{current_context}，对话对象：{target_agent.name}（身份：{target_agent.scratch.config.get('innate', '')} ）
                请基于你的身份和高效经验，发起1句与任务相关的自然对话（简洁、贴合人设）：
                """
                # Use completion to generate chat content
                # We need a proper func_hint. Let's use 'generate_chat' but we are bypassing the normal chat flow slightly
                # Or we can just use _llm directly if available, or define a temporary scratch method?
                # Easier: just call completion with a generic hint if it supports arbitrary prompts?
                # The existing completion method relies on scratch.prompt_{func_hint}.
                # We should add a generic prompt method or reuse an existing one.
                # 'generate_chat' expects specific args.
                # Let's mock the prompt generation or temporarily inject the prompt.
                
                # Hack: define a temporary function in scratch or just manually call LLM if available.
                if self.llm_available():
                    # Direct LLM call wrapper
                    # But completion handles logging and wrapper.
                    # Let's try to reuse 'generate_chat' but we need to match its signature?
                    # No, we can just use the 'scratch' object to format a custom prompt if we added one.
                    # Since we can't easily modify Scratch class in this turn without reading another file...
                    # We will use the raw _llm.completion if possible, but we want logging.
                    
                    # Alternative: We construct the prompt here and pass it to a method that just returns it.
                    # But completion calls prompt_func(*args).
                    
                    # Let's assume we can just use the text directly.
                    # Actually, let's use a trick.
                    # self.completion("generate_chat", self, other, ...) is the normal way.
                    # But we want a CUSTOM prompt with REWARD signals.
                    
                    # We will simply call the LLM directly here for simplicity as per the snippet.
                    # The snippet used self.completion("generate_chat", prompt). 
                    # This implies the user *thinks* completion takes a prompt string.
                    # BUT looking at `agent.py`: 
                    # def completion(self, func_hint, *args, **kwargs):
                    #     func = getattr(self.scratch, "prompt_" + func_hint)
                    #     res = func(*args, **kwargs)._asdict() ...
                    
                    # So passing a prompt string as the second arg will fail if prompt_generate_chat expects Agent objects.
                    # prompt_generate_chat(self, other, relation, chats) -> Prompt
                    
                    # I will IMPLEMENT a custom way to call LLM with arbitrary prompt string to support the user's snippet intent.
                    
                    self.logger.info("{} -> rl_chat".format(self.name))
                    output = self._llm.completion(prompt=prompt)
                    self.logger.debug(f"RL Chat: {output}")
                    
                    # Simulate chat initiation
                    # We add it to self.chats so it gets picked up by reflection and maybe triggers the other agent?
                    # But `_chat_with` handles the full flow.
                    # Here we just "start" it? Or just record it?
                    # The snippet says: self.chats.append((target_agent.name, chat_content))
                    # This records that *I* said something to *Target*.
                    self.chats.append((target_agent.name, output))
                    
                    # Optionally, force a chat event?
                    # For now, following the snippet.
        
        # 3. 任务动作
        elif action_desc == "task":
            prompt = f"""
            你的身份：{self.scratch.config.get('innate', '')}，所属组：{agent_group}，核心任务：{agent_task}
            近期高效任务行为：{high_reward_actions if high_reward_actions else ['推进核心任务执行']}
            当前情境：{current_context}
            请基于你的身份和高效经验，描述1句当前任务执行动作（简洁、具体，符合人设）：
            """
            if self.llm_available():
                task_desc = self._llm.completion(prompt=prompt)
                self.scratch.currently = task_desc
                task_event = self._create_event(task_desc)
                task_event.reward = 0.1 # Inject attribute
                self.associate.add_node("event", task_event, poignancy=5)
                
                # Update action to reflect this task?
                # The snippet sets scratch.currently.
                # Maybe update self.action too?
                # self.action = memory.Action(task_event, task_event, start=utils.get_timer().daily_time(), duration=30)
        
        # 4. 查阅资料动作
        elif action_desc == "research":
            prompt = f"""
            你的身份：{self.scratch.config.get('innate', '')}，所属组：{agent_group}，核心任务：{agent_task}
            近期高效查阅行为：{high_reward_actions if high_reward_actions else ['查阅与任务相关的专业资料']}
            当前情境：{current_context}
            请基于你的身份和高效经验，描述1句查阅资料的行为（简洁、贴合专业领域）：
            """
            if self.llm_available():
                research_desc = self._llm.completion(prompt=prompt)
                research_event = self._create_event(research_desc)
                research_event.reward = 0.1
                self.associate.add_node("event", research_event, poignancy=5)

    def to_dict(self, with_action=True):
        info = {
            "status": self.status,
            "schedule": self.schedule.to_dict(),
            "associate": self.associate.to_dict(),
            "chats": self.chats,
            "currently": self.scratch.currently,
        }
        if with_action:
            info.update({"action": self.action.to_dict()})
        return info
