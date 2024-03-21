from multiprocessing import Pipe
from copy import deepcopy
from ParallelPool.Worker import Worker


# Works in the main process and manages sub-workers
class PoolManager(object):
    def __init__(self, worker_num: int, env_params, policy="Random"):
        self.worker_num = worker_num
        self.env_params = env_params
        self.policy = policy

        # Buffer for workers and pipes
        self.workers = []
        self.pipes = []

        # Initialize workers
        for worker_idx in range(worker_num):
            parent_pipe, child_pipe = Pipe()
            self.pipes.append(parent_pipe)

            worker = Worker(
                pipe=child_pipe
            )
            self.workers.append(worker)

        # Start workers
        for worker in self.workers:
            worker.start()

        # Worker status: 0 for idle, 1 for busy
        self.worker_status = [0 for _ in range(worker_num)]

    def has_idle_server(self):
        for status in self.worker_status:
            if status == 0:
                return True

        return False

    def server_occupied_rate(self):
        occupied_count = 0.0

        for status in self.worker_status:
            occupied_count += status

        return occupied_count / self.worker_num

    def find_idle_worker(self):
        for idx, status in enumerate(self.worker_status):
            if status == 0:
                self.worker_status[idx] = 1
                return idx

        return None

    def assign_expansion_task(self, checkpoint_data, curr_node, saving_idx, task_simulation_idx):
        worker_idx = self.find_idle_worker()

        self.send_safe_protocol(worker_idx, "Expansion", (
            checkpoint_data,
            curr_node,
            saving_idx,
            task_simulation_idx
        ))

        self.worker_status[worker_idx] = 1

    def assign_simulation_task(self, sim_idx, node):
        worker_idx = self.find_idle_worker()

        self.send_safe_protocol(worker_idx, "Simulation", (
            sim_idx,
            node
        ))

        self.worker_status[worker_idx] = 1

    def get_complete_expansion_task(self):
        flag = False
        selected_worker_idx = -1

        while not flag:
            for worker_idx in range(self.worker_num):
                item = self.receive_safe_protocol_tapcheck(worker_idx)

                if item is not None:
                    flag = True
                    selected_worker_idx = worker_idx
                    break

        command, args = item
        assert command == "ReturnExpansion"

        # Set to idle
        self.worker_status[selected_worker_idx] = 0

        return args

    def get_complete_simulation_task(self):
        flag = False
        selected_worker_idx = -1

        while not flag:
            for worker_idx in range(self.worker_num):
                item = self.receive_safe_protocol_tapcheck(worker_idx)

                if item is not None:
                    flag = True
                    selected_worker_idx = worker_idx
                    break

        command, args = item
        assert command == "ReturnSimulation"

        # Set to idle
        self.worker_status[selected_worker_idx] = 0

        return args

    def send_safe_protocol(self, worker_idx, command, args):
        success = False

        while not success:
            self.pipes[worker_idx].send((command, args))

            ret = self.pipes[worker_idx].recv()
            if ret == command:
                success = True

    def wait_until_all_envs_idle(self):
        for worker_idx in range(self.worker_num):
            if self.worker_status[worker_idx] == 0:
                continue

            self.receive_safe_protocol(worker_idx)

            self.worker_status[worker_idx] = 0

    def receive_safe_protocol(self, worker_idx):
        """
        Receive message from pipe, it will block until message is received
        """
        self.pipes[worker_idx].poll(None)

        command, args = self.pipes[worker_idx].recv()

        self.pipes[worker_idx].send(command)

        return deepcopy(command), deepcopy(args)

    def receive_safe_protocol_tapcheck(self, worker_idx):
        """
        Receive message from pipe, but do not block
        """
        flag = self.pipes[worker_idx].poll()
        if not flag:
            return None

        command, args = self.pipes[worker_idx].recv()

        self.pipes[worker_idx].send(command)

        return deepcopy(command), deepcopy(args)

    def close_pool(self):
        for worker_idx in range(self.worker_num):
            self.send_safe_protocol(worker_idx, "KillProc", None)

        for worker in self.workers:
            worker.join()
