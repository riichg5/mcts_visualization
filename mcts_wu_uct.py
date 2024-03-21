# -*- coding: utf-8 -*-
# 基于多进程的并行mcts算法
import random
import sys
import math
import time
import uuid
from multiprocessing import Process, Queue, Pool
from ParallelPool.PoolManager import PoolManager
from copy import deepcopy
# from utils import is_array_in_list
from utils import get_child_nodes_color
from game import Game
import numpy as np


class Node(object):
    """
    蒙特卡罗树搜索的树结构的 Node，包含了父节点和子节点等信息，还有用于计算 UCB 的遍历次数和 quality 值，
    还有游戏选择这个 Node 的 State。
    """
    def __init__(self, state, action_to_state: int):
        self.parent = None
        self.uuid = uuid.uuid4()
        # expand的子节点
        self.children: list[Node] = []
        self.visit_times: int = 0
        # wu-uct
        self.completed_visit_times: int = 0
        self.quality_value = 0.0
        self.state = state
        self.action_to_state = action_to_state

        # 绘制MC树时需要的变量
        self.depth = None
        self.node_color = None
        self.is_root_node = False
        self.child_node_colors = None

    def set_state(self, state):
        self.state = deepcopy(state)

    def get_state(self):
        return deepcopy(self.state)

    def set_action_obj_id(self, action_to_state):
        self.action_to_state = action_to_state

    def get_action_obj_id(self):
        return self.action_to_state

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_completed_visit_times(self) -> int:
        return self.completed_visit_times

    def completed_visit_times_add_one(self):
        self.completed_visit_times += 1

    def get_visit_times(self):
        return self.visit_times

    def set_visit_times(self, times):
        self.visit_times = times

    def visit_times_add_one(self):
        self.visit_times += 1

    def get_quality_value(self):
        return self.quality_value

    def set_quality_value(self, value):
        self.quality_value = value

    def quality_value_add_n(self, n):
        self.quality_value += n

    def is_all_expand(self):
        """
        判断当前节点，是否将所有子节点都已经 expand(探索) 了
        """
        return len(self.children) == len(self.state['legal_actions'])

    def get_children(self):
        # 获得expand子节点
        return self.children

    def add_child(self, sub_node):
        # 添加expand子节点
        sub_node.set_parent(self)
        self.children.append(sub_node)

    # 绘制MC树需要的属性
    def set_depth(self, depth: int):
        """
        设置节点的深度, 值从0开始
        """
        self.depth = depth

    def get_depth(self) -> int:
        """
        获取节点深度，值从0开始
        """
        return self.depth

    def set_node_color(self, color: tuple):
        self.node_color = color

    def get_node_color(self):
        """
        返回当前节点的rgb颜色
        """
        return self.node_color

    def set_child_nodes_color(self):
        if self.is_root_node:
            self.child_node_colors = get_child_nodes_color(len(self.state['legal_actions']))

    def get_child_nodes_color(self):
        return self.child_node_colors

    def __repr__(self):
        return "Node: {}, Q/N: {}/{}, state: {}".format(
            hash(self), self.quality_value, self.visit_times, self.state)


class Mcts(object):
    def __init__(self):
        self.env_model: Game = None
        self.root_node = Node(None, None)
        self.nodes = []                         # 用于绘制MC树
        self.rest_rollout_times = 0
        # self.expansion_worker_pool = PoolManager(worker_num=10, env_params=None, policy="Random")
        self.simulation_worker_pool = PoolManager(worker_num=10, env_params=None, policy="Random")

        # 任务记录
        self.expansion_task_recorder = dict()
        self.unscheduled_expansion_tasks = list()
        self.simulation_task_recorder = dict()
        self.unscheduled_simulation_tasks = list()

    def set_env_model(self, env_model_object: Game):
        self.env_model = env_model_object

    def create_root_node(self, state):
        self.root_node = Node(deepcopy(state), None)
        self.root_node.set_node_color((0, 0, 0))
        self.root_node.set_depth(0)
        self.root_node.is_root_node = True
        self.root_node.set_child_nodes_color()
        self.update_nodes_list(self.root_node)

    def update_nodes_list(self, node):
        """
        使用nodes记录节点的深度和节点本身，用于绘制MC树
        数据格式为：
        [[第一层节点], [第二层节点], [第三层节点], ...]
        [[node], [node, node], [node, node, node], ...]
        """
        if node.get_depth() >= len(self.nodes):
            self.nodes.append([])
        self.nodes[node.get_depth()].append(node)

    def clear_nodes_list(self):
        self.nodes = []

    def clear_root_node(self):
        self.root_node = None

    def reset_rollout_times(self):
        self.rest_rollout_times = 150   # get_config("mcts")['rollout_times']

    def create_new_tree(self, state):
        self.clear_nodes_list()
        self.create_root_node(state)
        self.reset_rollout_times()

    def find_parent_node_in_tree(self, uuid_list: list):
        """
        从根节点开始，根据action_obj_id_list找到对应的节点
        """
        node = self.root_node
        # 如果是root节点，是没有父节点的
        if len(uuid_list) == 0:
            return None

        # root下面一个节点
        if len(uuid_list) == 1:
            return self.root_node

        for uid in uuid_list[0: len(uuid_list) - 1]:
            for child in node.get_children():
                if child.uuid == uid:
                    node = child
                    break
        return node

    def step(self, state):
        """
        将rollout次数用完，并返回下一步怎么走
        :param state:
        :return:
        """
        # self.expansion_worker_pool.wait_until_all_envs_idle()
        self.simulation_worker_pool.wait_until_all_envs_idle()

        if (self.root_node.get_state() is None
                or list(self.root_node.get_state()['obs'].flatten()) != list(state['obs'].flatten())):
            self.create_new_tree(state)

        self.rollout()

        return self.best_child(self.root_node, False).get_action_obj_id()

    def rollout(self):
        """
        使用多进程机制进行rollout，参考wu-uct算法
        """
        simulation_res_map: dict = dict()                   # 存放simulation结果的字典
        sim_idx: int = 0

        while self.rest_rollout_times > 0:
            if self.simulation_worker_pool.server_occupied_rate() < 0.99:
                sim_idx += 1
                sim_idx, uuid_list, expand_node = self.tree_policy(self.root_node, sim_idx)
                print("sim_idx: %s, uuid_list: %s" % (sim_idx, uuid_list))
                # 根据action_obj_id_list找到root_node下面对应的节点
                # parent_node = self.find_parent_node_in_tree(uuid_list)
                # parent_node.add_child(expand_node)
                # self.update_nodes_list(expand_node)
                simulation_res_map[sim_idx] = {
                    "sim_idx": sim_idx,
                    "expand_node": expand_node,
                    "simulation_reward": None,
                    "simulation_state": None
                }
                self.incomplete_backup(expand_node)
                self.unscheduled_simulation_tasks.append(sim_idx)
                self.simulation_task_recorder[sim_idx] = expand_node

            while len(self.unscheduled_simulation_tasks) > 0 and self.simulation_worker_pool.has_idle_server():
                idx = np.random.randint(0, len(self.unscheduled_simulation_tasks))
                task_idx = self.unscheduled_simulation_tasks.pop(idx)

                self.simulation_worker_pool.assign_simulation_task(
                    task_idx,
                    self.simulation_task_recorder[task_idx]
                )

            if self.simulation_worker_pool.server_occupied_rate() > 0.99:
                task_idx, reward = self.simulation_worker_pool.get_complete_simulation_task()
                simulation_res = simulation_res_map[task_idx]
                simulation_res['simulation_reward'] = reward
                simulation_res['simulation_state'] = "done"
                self.rest_rollout_times -= 1
                print("rest_rollout_times: %s" % self.rest_rollout_times)
                self.complete_backup(simulation_res['expand_node'], reward)

    def tree_policy(self, root_node: Node, sim_idx: int) -> tuple:
        """
        蒙特卡罗树搜索的 Selection 和 Expansion 阶段，传入当前需要开始搜索的节点（例如根节点），
        根据 exploration/exploitation 算法返回最好的需要 expend 的节点，注意如果节点是叶子结点直接返回。
        基本策略是先找当前未选择过的子节点，如果有多个则随机选。如果都选择过就找权衡过 exploration/exploitation 的 UCB 值最大的，
        如果 UCB 值相等则随机选。
        """
        node = root_node
        uuid_list = []
        # Check if the current node is the leaf node
        while node.get_state()['legal_actions']:
            if node.is_all_expand():
                # 如果所有节点都探索过了，则选择最好的节点
                temp_node = self.best_child(node, True)
                if temp_node is None:
                    node = random.choice(node.get_children())
                else:
                    node = temp_node
                uuid_list.append(node.uuid)
            else:
                sub_node = self.expand(node)
                uuid_list.append(sub_node.uuid)
                return sim_idx, uuid_list, sub_node
        return sim_idx, uuid_list, node

    def expand(self, node: Node) -> Node:
        """
        输入一个节点，在该节点上拓展一个新的节点，使用 random 方法执行 Action，返回新增的节点。
        注意，需要保证新增的节点与其他节点 Action 不同。
        """
        tried_sub_actions = [
            sub_node.get_action_obj_id() for sub_node in node.get_children()
        ]
        print('expansion...')
        # print('state of children nodes: {}'.format(tried_sub_node_states))

        action_obj_id = random.choice(node.state['legal_actions'])
        while action_obj_id in tried_sub_actions:
            action_obj_id = random.choice(node.state['legal_actions'])

        self.env_model.set_state(node.get_state())
        state, action, next_state, reward, done = self.env_model.step(action_obj_id)

        sub_node = Node(next_state, action)
        node.add_child(sub_node)

        sub_node.set_depth(node.get_depth() + 1)
        if sub_node.get_depth() == 1:
            # 如果是第一层节点, 设置节点颜色事先已经初始化的颜色列表
            sub_node.set_node_color(node.get_child_nodes_color().pop(0))
        else:
            # 子节点的颜色与父节点相同
            sub_node.set_node_color(node.get_node_color())

        self.update_nodes_list(sub_node)
        return sub_node

    def simulation(self, expand_no: int, node: Node, simulation_queue: Queue) -> None:
        """
        蒙特卡罗树搜索的 Simulation 阶段，输入一个需要 expand 的节点，随机操作后创建新的节点，返回新增节点的 reward。
        注意输入的节点应该不是子节点，而且是有未执行的 Action可以 expend 的。
        基本策略是随机选择Action。
        """
        print('simulation...')
        # Get the state of the game
        current_state = deepcopy(node.get_state())

        # Run until the game over
        while current_state['legal_actions'] != []:
            # Pick one random action to play and get next state
            self.env_model.set_state(current_state)
            state, action, next_state, reward, done = self.env_model.random_step()
            current_state = next_state
            if done:
                simulation_queue.put((expand_no, reward))
                return
                # return reward

        # If node is leaf
        simulation_queue.put((expand_no, 0))
        # return 0

    @staticmethod
    def best_child(root_node: Node, is_exploration: bool) -> Node:
        """
        使用 UCB 算法，权衡 exploration 和 exploitation 后选择得分最高的子节点，注意如果是预测阶段直接选择当前Q值得分最高的。
        """
        print('select best child, is_exploration: %s...' % is_exploration)
        # Use the min float value
        best_score = -sys.maxsize
        best_sub_node = None

        # Travel all sub nodes to find the best one
        for sub_node in root_node.get_children():
            # 如果暂时还没有做simulation，则先忽略
            if sub_node.get_completed_visit_times() == 0:
                continue
            # print(sub_node)
            # Ignore exploration for inference
            if is_exploration:
                # C = 1 / math.sqrt(2.0)
                C = 2       # 这个值要调试得到
            else:
                C = 0.0

            # UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
            left = sub_node.get_quality_value() / (sub_node.get_completed_visit_times())
            # right = 2.0 * math.log(node.get_visit_times()) / sub_node.get_visit_times()
            # 此公式参考快手的wu-uct算法论文
            right = (
                        2 * math.log(root_node.get_visit_times())
                        / (sub_node.get_visit_times())
                    )
            score = left + C * math.sqrt(right)

            if score > best_score:
                best_sub_node = sub_node
                best_score = score

        return best_sub_node


    @staticmethod
    def incomplete_backup(node: Node):
        """
        蒙特卡洛树搜索的 Backpropagation 阶段，输入前面获取需要 expend 的节点和新执行 Action 的 reward，
        反馈给 expend 节点和上游所有节点并更新对应数据。
        """
        # Update util the root node
        while node is not None:
            # Update the visit times
            node.visit_times_add_one()

            # Change the node to the parent node
            node = node.parent

    @staticmethod
    def complete_backup(node: Node, reward: int):
        """
        反向传播完全节点完成simulation次数
        """
        while node is not None:
            node.completed_visit_times_add_one()
            # Update the quality value
            node.quality_value_add_n(reward)
            node = node.parent


