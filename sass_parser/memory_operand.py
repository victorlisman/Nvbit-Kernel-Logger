
import re

class MemoryOperand:
    def __init__(self, raw: str):
        self.raw = raw
        self.base_register = None
        self.offset_register = None
        self.constant_offset = 0
        self.constant_memory = None
        self.scale = 1
        self.parse_operand()
    
    def parse_operand(self):
        operand = self.raw.strip('[]')
        
        const_match = re.match(r'c\[([^\]]+)\]\[([^\]]+)\]', operand)
        if const_match:
            self.constant_memory = {
                'bank': const_match.group(1),
                'offset': const_match.group(2)
            }
            return
        
        if '+' in operand:
            parts = operand.split('+')
            self.base_register = parts[0].strip()
            
            for part in parts[1:]:
                part = part.strip()
                if part.startswith('0x') or part.isdigit():
                    self.constant_offset += int(part, 16 if part.startswith('0x') else 10)
                elif part.startswith('R'):
                    self.offset_register = part
        else:
            if operand.startswith('R'):
                self.base_register = operand
            elif operand.startswith('0x') or operand.isdigit():
                self.constant_offset = int(operand, 16 if operand.startswith('0x') else 10)
    
    def get_symbolic_expression(self) -> str:
        if self.constant_memory:
            return f"c[{self.constant_memory['bank']}][{self.constant_memory['offset']}]"
        
        expr_parts = []
        if self.base_register:
            expr_parts.append(self.base_register)
        if self.offset_register:
            expr_parts.append(self.offset_register)
        if self.constant_offset != 0:
            expr_parts.append(f"{self.constant_offset:#x}")
        
        return " + ".join(expr_parts) if expr_parts else "0"
    
    def __repr__(self):
        return f"MemoryOperand({self.raw} -> {self.get_symbolic_expression()})"
