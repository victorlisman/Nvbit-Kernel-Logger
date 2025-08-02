from typing import List, Optional, Dict
try:
    from .memory_operand import MemoryOperand
except ImportError:
    from memory_operand import MemoryOperand

class SassInstruction:
    def __init__(self, predicate: Optional[str], opcode: str, operands: List[str], raw: str):
        self.predicate = predicate or ""
        self.opcode = opcode
        self.operands = operands
        self.raw = raw
        self.memory_operands = []
        self._parse_memory_operands()

    def _parse_memory_operands(self):
        for i, operand in enumerate(self.operands):
            if '[' in operand and ']' in operand:
                self.memory_operands.append(MemoryOperand(operand))

    def __repr__(self):
        pred = f"{self.predicate} " if self.predicate else ""
        return f"{pred}{self.opcode} {', '.join(self.operands)}"
    
    def is_memory_instruction(self) -> bool:
        memory_opcodes = {'LDG', 'STG', 'LDS', 'STS', 'LDL', 'STL', 'LDGSTS', 'LDSLK', 'STSCUL', 'LDC'}
        return any(op in self.opcode.split('.')[0] for op in memory_opcodes)
    
    def get_access_size(self) -> int:
        size_map = {
            '.8': 1, '.16': 2, '.32': 4, '.64': 8, '.128': 16,
            '.U8': 1, '.U16': 2, '.U32': 4, '.U64': 8,
            '.S8': 1, '.S16': 2, '.S32': 4, '.S64': 8,
            '.F32': 4, '.F64': 8
        }
        
        for suffix, size in size_map.items():
            if suffix in self.opcode:
                return size
        
        if '.S16' in self.opcode or '.U16' in self.opcode:
            return 2
        elif '.S8' in self.opcode or '.U8' in self.opcode:
            return 1
        elif '.S32' in self.opcode or '.U32' in self.opcode or '.F32' in self.opcode:
            return 4
        elif '.S64' in self.opcode or '.U64' in self.opcode or '.F64' in self.opcode:
            return 8
        
        return 4  
    
    def get_memory_access_info(self) -> Optional[Dict]:
        if not self.is_memory_instruction():
            return None
            
        access_info = {
            'type': 'load' if self.opcode.startswith('LD') else 'store',
            'scope': self._get_memory_scope(),
            'operands': self.operands,
            'raw': self.raw,
            'access_size': self.get_access_size(),
            'memory_operands': [],
            'addressing_expressions': []
        }
        
        for mem_op in self.memory_operands:
            access_info['memory_operands'].append({
                'raw': mem_op.raw,
                'symbolic_expression': mem_op.get_symbolic_expression(),
                'base_register': mem_op.base_register,
                'offset_register': mem_op.offset_register,
                'constant_offset': mem_op.constant_offset,
                'constant_memory': mem_op.constant_memory
            })
            access_info['addressing_expressions'].append(mem_op.get_symbolic_expression())
                
        return access_info
    
    def _get_memory_scope(self) -> str:
        if 'G' in self.opcode:
            return 'global'
        elif 'S' in self.opcode:
            return 'shared'  
        elif 'L' in self.opcode:
            return 'local'
        elif 'C' in self.opcode:
            return 'constant'
        else:
            return 'unknown'
