# -*- coding: utf-8 -*-
import sys
import math
import random
import numpy as np

from copy import deepcopy
from utils import is_array_in_list
from utils import get_child_nodes_color

class Node(object):
    """
    蒙特卡罗树搜索的树结构的 Node，包含了父节点和子节点等信息，还有用于计算 UCB 的遍历次数和 quality 值，
    还有游戏选择这个 Node 的 State。
    """
    def __init__(self, state, action_to_state):
        self.parent = None
        self.children = []
        self.visit_times = 0
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

    def set_action_to_state(self, action_to_state):
        self.action_to_state = action_to_state

    def get_action_to_state(self):
        return self.action_to_state

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_children(self):
        return self.children

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

    def add_child(self, sub_node):
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

    def set_node_color(self, color):
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
        self.env_model = None
        self.root_node = Node(None, None)
        self.nodes = []
        self.rest_rollout_times = 0

    def set_env_model(self, env_model_object):
        self.env_model = env_model_object

    def create_root_node(self, state):
        self.root_node = Node(deepcopy(state), None)
        self.root_node.set_node_color((0,0,0))
        self.root_node.set_depth(0)
        self.root_node.is_root_node = True
        self.root_node.set_child_nodes_color()

        self.update_nodes_list(self.root_node)

    def update_nodes_list(self, node):
        """
        记录节点的深度和节点本身，用于绘制MC树
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
        self.rest_rollout_times = 150

    def create_new_tree(self, state):
        self.clear_nodes_list()
        self.create_root_node(state)
        self.reset_rollout_times()

    def main(self, state):
      """
      实现蒙特卡洛树搜索算法，传入一个根节点，在有限的时间内根据之前已经探索过的树结构 expand 新节点和更新数据，
      然后返回只要 exploitation 最高的子节点。
      蒙特卡洛树搜索包含四个步骤，Selection、Expansion、Simulation、Backpropagation。
      前两步使用tree policy找到值得探索的节点。
      第三步使用default policy也就是在选中的节点上随机算法选一个子节点并计算reward。
      最后一步使用backup也就是把reward更新到所有经过的选中节点的节点上。
      进行预测时，只需要根据Q值选择exploitation最大的节点即可，找到下一个最优的节点。
      """
      computation_budget = 1000
      root_node = Node(deepcopy(state), None)

      # Run as much as possible under the computation budget
      for i in range(computation_budget):
          print('rollout: {}'.format(i))

          # 1. Find the best node to expand
          expand_node = self.tree_policy(root_node)

          # 2. Random run to add node and get reward
          reward = self.default_policy(expand_node)

          # 3. Update all passing nodes with reward
          self.backup(expand_node, reward)

      # N. Get the best next node
      best_next_node = self.best_child(root_node, False)

      return best_next_node.get_action_to_state()

    def step(self, state):
        if self.root_node.get_state() is None:
            self.create_new_tree(state)
        else:
            if list(self.root_node.get_state()['obs'].flatten()) != list(state['obs'].flatten()): # 即根节点更新时，更新整一颗树
                self.create_new_tree(state)
            else:
                self.rollout()

        if self.rest_rollout_times == 0:
            return self.best_child(self.root_node, False).get_action_to_state()

    def rollout(self):
        print('rest rollout times: {}'.format(self.rest_rollout_times))
        # 1. Find the best node to expand
        expand_node = self.tree_policy(self.root_node)

        # 2. Random run to add node and get reward
        reward = self.default_policy(expand_node)

        # 3. Update all passing nodes with reward
        self.backup(expand_node, reward)

        self.rest_rollout_times -= 1

    def tree_policy(self, node: Node) -> Node:

        """
        蒙特卡罗树搜索的 Selection 和 Expansion 阶段，传入当前需要开始搜索的节点（例如根节点），
        根据 exploration/exploitation 算法返回最好的需要 expend 的节点，注意如果节点是叶子结点直接返回。
        基本策略是先找当前未选择过的子节点，如果有多个则随机选。如果都选择过就找权衡过 exploration/exploitation 的 UCB 值最大的，
        如果 UCB 值相等则随机选。
        """
        # Check if the current node is the leaf node
        while node.get_state()['legal_actions'] != []:
            if node.is_all_expand():
                # 如果所有节点都探索过了，则选择最好的节点
                node = self.best_child(node, True)
            else:
                # Return the new sub node
                sub_node = self.expand(node)
                return sub_node

        # Return the leaf node
        return node

    def expand(self, node: Node):
        """
        输入一个节点，在该节点上拓展一个新的节点，使用 random 方法执行 Action，返回新增的节点。
        注意，需要保证新增的节点与其他节点 Action 不同。
        """
        tried_sub_node_states = [
            sub_node.get_state()['obs'] for sub_node in node.get_children()
        ]
        print('expansion...')
        # print('state of children nodes: {}'.format(tried_sub_node_states))

        self.env_model.set_state(node.get_state())
        state, action, next_state, reward, done = self.env_model.random_step()

        # Check until get the new state which has the different action from others
        while is_array_in_list(next_state['obs'], tried_sub_node_states):
            self.env_model.set_state(node.get_state())
            state, action, next_state, reward, done = self.env_model.random_step()

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


    def default_policy(self, node):
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
                return reward

        # If node is leaf
        return 0

    def best_child(self, node: Node, is_exploration: bool) -> Node:
      """
      使用 UCB 算法，权衡 exploration 和 exploitation 后选择得分最高的子节点，注意如果是预测阶段直接选择当前Q值得分最高的。
      """
      print('select best child, is_exploration: %s...' % is_exploration)
      # Use the min float value
      best_score = -sys.maxsize
      best_sub_node = None

      # Travel all sub nodes to find the best one
      for sub_node in node.get_children():
        # print(sub_node)
        # Ignore exploration for inference
        if is_exploration:
          # C = 1 / math.sqrt(2.0)
          C = 2
        else:
          C = 0.0

        # UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
        left = sub_node.get_quality_value() / sub_node.get_visit_times()
        # right = 2.0 * math.log(node.get_visit_times()) / sub_node.get_visit_times()
        right = math.log(node.get_visit_times()) / sub_node.get_visit_times()
        score = left + C * math.sqrt(right)

        if score > best_score:
          best_sub_node = sub_node
          best_score = score

      return best_sub_node

    def backup(self, node: Node, reward: int):
      """
      蒙特卡洛树搜索的 Backpropagation 阶段，输入前面获取需要 expend 的节点和新执行 Action 的 reward，
      反馈给 expend 节点和上游所有节点并更新对应数据。
      """
      # Update util the root node
      while node != None:
        # Update the visit times
        node.visit_times_add_one()

        # Update the quality value
        node.quality_value_add_n(reward)

        # Change the node to the parent node
        node = node.parent





























