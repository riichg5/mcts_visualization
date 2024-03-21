from multiprocessing import Process
from copy import deepcopy
from game import Game


# Slave workers
class Worker(Process):
    def __init__(self, pipe):
        super(Worker, self).__init__()
        self.env_model = Game()
        self.pipe = pipe

    def run(self):
        print("> Worker ready.")

        while True:
            # Wait for tasks
            command, args = self.receive_safe_protocol()

            if command == "KillProc":
                return
            elif command == "Expansion":
                checkpoint_data, curr_node, saving_idx, task_idx = args

                # Select expand action, and do expansion
                expand_action, next_state, reward, done, \
                    checkpoint_data = self.expand_node(checkpoint_data, curr_node)

                item = (expand_action, next_state, reward, done, checkpoint_data,
                        saving_idx, task_idx)

                self.send_safe_protocol("ReturnExpansion", item)
            elif command == "Simulation":
                if args is None:
                    raise RuntimeError
                else:
                    sim_idx, node = args
                    sim_idx, accu_reward = self.simulate(sim_idx, node)

                self.send_safe_protocol("ReturnSimulation", (sim_idx, accu_reward))

    def expand_node(self, checkpoint_data, curr_node):
        self.wrapped_env.restore(checkpoint_data)

        # Choose action to expand, according to the shallow copy node
        expand_action = curr_node.select_expand_action()

        # Execute the action, and observe new state, etc.
        next_state, reward, done = self.wrapped_env.step(expand_action)

        if not done:
            checkpoint_data = self.wrapped_env.checkpoint()
        else:
            checkpoint_data = None

        return expand_action, next_state, reward, done, checkpoint_data

    def simulate(self, sim_idx: int, node):
        """
        蒙特卡罗树搜索的 Simulation 阶段，输入一个需要 expand 的节点，随机操作后创建新的节点，返回新增节点的 reward。
        注意输入的节点应该不是子节点，而且是有未执行的 Action可以 expend 的。
        基本策略是随机选择Action。
        """
        print('simulation...')
        # Get the state of the game
        current_state = deepcopy(node.get_state())

        # Run until the game over
        while current_state['legal_actions']:
            # Pick one random action to play and get next state
            self.env_model.set_state(current_state)
            state, action, next_state, reward, done = self.env_model.random_step()
            current_state = next_state
            if done:
                return sim_idx, reward
                # return reward

        # If node is leaf
        return sim_idx, 0

    # Send message through pipe
    def send_safe_protocol(self, command, args):
        success = False

        count = 0
        while not success:
            self.pipe.send((command, args))

            ret = self.pipe.recv()
            if ret == command or count >= 10:
                success = True

            count += 1

    # Receive message from pipe
    def receive_safe_protocol(self):
        self.pipe.poll(None)

        command, args = self.pipe.recv()

        self.pipe.send(command)

        return deepcopy(command), deepcopy(args)
