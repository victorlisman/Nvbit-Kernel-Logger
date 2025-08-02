import re
from typing import List, Dict, Optional

try:
    from .sass_instruction import SassInstruction
except ImportError:
    from sass_instruction import SassInstruction

def parse_sass_line(line: str) -> Optional[SassInstruction]:
    line = line.strip().rstrip(';')
    if not line or line.startswith("NOP") or line.startswith("BRA"):
        return None

    predicate = ""
    if line.startswith("@"):
        pred_match = re.match(r"(@[\!A-Z0-9]+)\s+", line)
        if pred_match:
            predicate = pred_match.group(1)
            line = line[pred_match.end():]

    opcode_match = re.match(r"([A-Z.]+)\s+(.*)", line)
    if not opcode_match:
        return None

    opcode = opcode_match.group(1)
    operands = [op.strip() for op in opcode_match.group(2).split(",")]

    return SassInstruction(predicate, opcode, operands, line)

def format_bytes(bytes_value: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.2f} TB"
