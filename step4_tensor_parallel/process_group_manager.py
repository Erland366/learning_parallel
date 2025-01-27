import os
import torch
import torch.distributed as dist

class ProcessGroupManager:
    def __init__(self, dp_size, pp_size, tp_size):
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.getenv("LOCAL_RANK", self.global_rank % self.world_size))

        assert self.world_size == dp_size * pp_size * tp_size, f"World size ({self.world_size}) != DP ({dp_size}) * PP ({pp_size}) * TP ({tp_size})"

        self.grid = torch.arange(self.world_size).view(dp_size, pp_size, tp_size) # DP * PP * TP grid
        # Find the position of the current rank in the grid
        self.dp_rank, self.pp_rank, self.tp_rank = (self.grid == self.global_rank).nonzero().squeeze(0).tolist()

        self.tp_group = dist.new_subgroups_by_enumeration([self.grid[d, p, :].tolist() for d in range(dp_size) for p in range(pp_size)])[0]
        self.pp_group = dist.new_subgroups_by_enumeration([self.grid[d, :, t].tolist() for d in range(dp_size) for t in range(tp_size)])[0]
        self.dp_group = dist.new_subgroups_by_enumeration([self.grid[:, p, t].tolist() for p in range(pp_size) for t in range(tp_size)])[0]
        self.pp_dp_group = dist.new_subgroups_by_enumeration([self.grid[:, :, t].flatten().tolist() for t in range(tp_size)])

        self.tp_group_ids = self.grid[self.dp_rank, self.pp_rank, :].tolist()
        self.pp_group_ids = self.grid[self.dp_rank, :, self.tp_rank].tolist()
        self.dp_group_ids = self.grid[:, self.pp_rank, self.tp_rank].tolist()

        self.tp_world_size = dist.get_world_size(group=self.tp_group) # For now I think this always will be the same with `tp_size`
        self.tp_first_rank = self.tp_group_ids[0]
        self.tp_last_rank = self.tp_group_ids[-1]

        self.pp_world_size = dist.get_world_size(group=self.pp_group) # It always be `pp_size` at least for now
        self.pp_first_rank = self.pp_group_ids[0]
        self.pp_last_rank = self.pp_group_ids[-1]
        self.pp_is_first_stage = self.pp_rank == 0
        self.pp_is_last_stage = self.pp_rank == self.pp_world_size - 1
        self.pp_next_rank = None if self.pp_rank == self.pp_world_size - 1 else int(self.grid[self.dp_rank, self.pp_rank + 1, self.tp_rank].item())
        self.pp_prev_rank = None if self.pp_rank == 0 else int(self.grid[self.dp_rank, self.pp_rank - 1, self.tp_rank].item())

        self.dp_world_size = dist.get_world_size(group=self.dp_group)
        self.dp_first_rank = self.dp_group_ids[0]
        self.dp_last_rank = self.dp_group_ids[-1]

    def __str__(self):
        return f"DP({self.dp_world_size})-PP({self.pp_world_size})-TP({self.tp_world_size})-Rank({self.global_rank})"


class MockProcessGroupManager:
    def __init__(self, dp_size, pp_size, tp_size, global_rank, world_size):
        self.global_rank = global_rank
        self.world_size = world_size
        # For single-GPU simulation, local_rank is the same as global_rank
        self.local_rank = global_rank  

        assert self.world_size == dp_size * pp_size * tp_size, f"World size ({self.world_size}) != DP ({dp_size}) * PP ({pp_size}) * TP ({tp_size})"

        self.grid = torch.arange(self.world_size).view(dp_size, pp_size, tp_size)  # DP * PP * TP grid
        self.dp_rank, self.pp_rank, self.tp_rank = (self.grid == self.global_rank).nonzero().flatten().tolist()

        # Tensor parallelism
        self.tp_group_ids = self.grid[self.dp_rank, self.pp_rank, :].tolist()
        self.tp_world_size = len(self.tp_group_ids)
        self.tp_first_rank = self.tp_group_ids[0]
        self.tp_last_rank = self.tp_group_ids[-1]

        # Pipeline parallelism
        self.pp_group_ids = self.grid[self.dp_rank, :, self.tp_rank].tolist()
        self.pp_world_size = len(self.pp_group_ids)
        self.pp_first_rank = self.pp_group_ids[0]
        self.pp_last_rank = self.pp_group_ids[-1]
        self.pp_is_first_stage = self.pp_rank == 0
        self.pp_is_last_stage = self.pp_rank == self.pp_world_size - 1
        self.pp_next_rank = None if self.pp_is_last_stage else self.grid[self.dp_rank, self.pp_rank + 1, self.tp_rank].item()
        self.pp_prev_rank = None if self.pp_is_first_stage else self.grid[self.dp_rank, self.pp_rank - 1, self.tp_rank].item()

        # Data parallelism
        self.dp_group_ids = self.grid[:, self.pp_rank, self.tp_rank].tolist()
        self.dp_world_size = len(self.dp_group_ids)
        self.dp_first_rank = self.dp_group_ids[0]
        self.dp_last_rank = self.dp_group_ids[-1]

        # PP x DP
        self.pp_dp_group_ids = self.grid[:, :, self.tp_rank].flatten().tolist()

    def __str__(self):
        return f"DP({self.dp_world_size})-PP({self.pp_world_size})-TP({self.tp_world_size})-Rank({self.global_rank})"

def setup_process_group_manager(dp_size, pp_size, tp_size):
    global process_group_manager
    process_group_manager = ProcessGroupManager(dp_size, pp_size, tp_size)

def main():
    dp_size = 2
    pp_size = 2
    tp_size = 8
    world_size = dp_size * pp_size * tp_size

    for global_rank in range(world_size):
        pg_manager = MockProcessGroupManager(dp_size, pp_size, tp_size, global_rank, world_size)
        print(pg_manager)
        print(f"  TP Group (Rank {global_rank}): {pg_manager.tp_group_ids} (Size: {pg_manager.tp_world_size})")
        print(f"  PP Group (Rank {global_rank}): {pg_manager.pp_group_ids} (Size: {pg_manager.pp_world_size})")
        print(f"  DP Group (Rank {global_rank}): {pg_manager.dp_group_ids} (Size: {pg_manager.dp_world_size})")
        print(f"  PP x DP Group (Rank {global_rank}): {pg_manager.pp_dp_group_ids} (Size: {len(pg_manager.pp_dp_group_ids)})")

if __name__ == "__main__":
    main()