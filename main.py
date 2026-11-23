"""Trae launch entry and coordinator API."""
import argparse
import json
import os
from pathlib import Path
from compare import Chat, write_vimax
from script_agent import BRIEF, generate

ROOT = Path(__file__).resolve().parent

def load_config():
    env = ROOT / '.env'
    if env.exists():
        for line in env.read_text(encoding='utf-8-sig').splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    for key in ('CHAT_API_KEY', 'CHAT_MODEL', 'CHAT_BASE_URL'):
        if not os.environ.get(key):
            raise ValueError(f'请在 .env 中设置 {key}')

def run_agent(idea, slots=None, backend='auto', budget='standard', chat=None):
    """budget is a routing preference, not a provider spending cap."""
    if backend not in ('auto','baseline','vimax') or budget not in ('economy','standard','ample'):
        raise ValueError('无效的后端或预算档位')
    if chat is None:
        load_config()
        chat = Chat()
    selected = None
    reason = ''
    def call(instruction, payload):
        nonlocal selected, reason
        if instruction == BRIEF:
            return chat.structured(BRIEF + ' 另返回 story_complexity 字段，仅当创意需要显著的人物动机、冲突或反转时填 high，否则填 normal。', payload)
        if backend != 'auto':
            selected, reason = backend, '手动选择'
        elif budget == 'ample' and payload['brief'].get('story_complexity') == 'high':
            selected, reason = 'vimax', '预算充足，且需求涉及复杂叙事'
        else:
            selected, reason = 'baseline', '当前预算或叙事需求适合基础编剧'
        if selected == 'vimax':
            return write_vimax(payload, chat)[0]
        return chat.structured(instruction,payload)
    result = generate(idea,call,slots)
    result['routing'] = dict(requested=backend,selected=selected,budget=budget,reason=reason)
    result['usage'] = dict(calls=chat.calls,reported_tokens=chat.tokens)
    return result

def main():
    p=argparse.ArgumentParser(description='灵感 → 可交给分镜 Agent 的结构化剧本')
    p.add_argument('idea',nargs='?')
    p.add_argument('--backend',choices=['auto','baseline','vimax'],default='auto')
    p.add_argument('--budget',choices=['economy','standard','ample'],default='standard')
    p.add_argument('--slots',type=int,nargs='+')
    p.add_argument('--demo',action='store_true',help='离线查看已有真实样例')
    args=p.parse_args()
    if args.demo:
        print((ROOT/'sample'/'baseline.json').read_text(encoding='utf-8'))
        return
    idea=args.idea or input('请输入视频灵感：')
    result=run_agent(idea,args.slots,args.backend,args.budget)
    from uuid import uuid4
    folder=ROOT/'outputs'
    folder.mkdir(exist_ok=True)
    path=folder/(uuid4().hex+'.json')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print('已保存',path)

if __name__=='__main__':
    main()
