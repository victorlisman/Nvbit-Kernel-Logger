from typing import List, Dict

class KernelInfo:
    def __init__(self, name: str, grid_dim: tuple, block_dim: tuple, args: List[Dict]):
        self.name = name
        self.grid_dim = grid_dim
        self.block_dim = block_dim
        self.args = args
        self.total_threads = grid_dim[0] * grid_dim[1] * grid_dim[2] * block_dim[0] * block_dim[1] * block_dim[2]
        
    def get_memory_args(self) -> List[Dict]:
        return [arg for arg in self.args if arg.get('type', '').endswith('*')]
