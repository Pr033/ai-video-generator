"""Public entry point for the coordinator."""
from compare import Chat, write_vimax
from script_agent import BRIEF, generate

def generate_script(idea, slots=None, backend='baseline'):
    if backend not in ('baseline', 'vimax'):
        raise ValueError('backend must be baseline or vimax')
    chat = Chat()
    def call(instruction, payload):
        if instruction == BRIEF or backend == 'baseline':
            return chat.structured(instruction, payload)
        script, _ = write_vimax(payload, chat)
        return script
    return generate(idea, call, slots)
