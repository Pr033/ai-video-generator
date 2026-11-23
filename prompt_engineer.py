"""Shot-level English prompt generation. Python 3.10+."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
from compare import Chat
from main import load_config

FIELDS = ('visual_description','action','start_state','end_state','camera','continuity_requirements','transition','duration_sec','characters','scene_id')
SYSTEM = """You translate or refine storyboard shots into standalone English prompts for a video renderer.
Return JSON {"prompts":[{"shot_id":"...","video_prompt":"..."}]} with one entry per supplied shot.
Preserve each shot's composition, action, duration, camera, opening and ending state, continuity, and shared style.
Apply must/avoid/sound constraints as relevant. Scope final-shot camera instructions to the final shot.
Preserve identity and staging. Never invent dialogue, props, new actions or a camera change.
For translate mode use a concise literal production description. For refine mode clarify blocking, framing and motion.
Return English prose. No commentary."""

def enhance_board(board, mode='auto', budget='standard', chat=None):
    if mode not in ('auto','template','translate','refine') or budget not in ('economy','standard','ample'):
        raise ValueError('无效模式或预算')
    result=deepcopy(board)
    if result.get('status')=='needs_input':
        return result
    shots=result.get('shots',[])
    if result.get('status')!='ready' or not shots:
        raise ValueError('需要已完成的分镜')
    ids=[s['shot_id'] for s in shots]
    if not all(isinstance(x,str) and x for x in ids) or len(set(ids))!=len(ids):
        raise ValueError('shot_id 缺失或重复')
    brief={k:result['brief'].get(k) for k in ('theme','style','sound','must','avoid','camera')}
    pending=[]
    for s in shots:
        context={k:s[k] for k in FIELDS}
        # Conservatively route non-ASCII content through the translation backend.
        data=json.dumps([context,brief],ensure_ascii=False)
        route=mode
        if route=='auto':
            if budget=='ample' and (s['transition']=='continue' or result['brief'].get('story_complexity')=='high'):
                route='refine'
            else:
                route='template' if data.isascii() else 'translate'
        if route=='template':
            if not data.isascii():
                raise ValueError('模板适用于英文输入，请使用 auto 或 translate')
            s['video_prompt']=' '.join([f"{k.replace('_',' ')}: {v}." for k,v in context.items() if v]+
                [f"{k}: {brief[k]}." for k in ('style','sound','must','avoid') if brief[k]])
        else:
            pending.append({'shot_id':s['shot_id'],'mode':route,'final_shot':s is shots[-1],**context})
        s['prompt_mode']=route
    if pending:
        if chat is None:
            load_config()
            chat=Chat()
        rows=chat.structured(SYSTEM,{'brief':brief,'shots':pending})['prompts']
        mapping={r['shot_id']:r['video_prompt'] for r in rows}
        if len(mapping)!=len(rows) or set(mapping)!={p['shot_id'] for p in pending}:
            raise ValueError('模型返回的镜头编号与输入不一致')
        for s in shots:
            if s['shot_id'] in mapping:
                prompt=mapping[s['shot_id']]
                if not isinstance(prompt,str) or not re.search('[A-Za-z]{2,}',prompt) or re.search('[\u3400-\u9fff]',prompt):
                    raise ValueError('视频提示词需为英文')
                s['video_prompt']=prompt.strip()
    return result

def enhance_prompt(shot, brief, **kwargs):
    """Single-shot convenience API; use enhance_board for sequence-wide constraints."""
    return enhance_board({'status':'ready','brief':brief,'shots':[shot]},**kwargs)['shots'][0]['video_prompt']

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('board',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mode',choices=['auto','template','translate','refine'],default='auto')
    parser.add_argument('--budget',choices=['economy','standard','ample'],default='standard')
    args=parser.parse_args()
    output=enhance_board(json.loads(args.board.read_text(encoding='utf-8')),args.mode,args.budget)
    args.out.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
