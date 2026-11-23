"""Import build_storyboard(script) or pass a saved script JSON to the CLI."""
import argparse
import asyncio
import json
from pathlib import Path
from main import load_config
from compare import Chat

PROMPT = """你是分镜导演。按给定剧本逐段拆镜，务必覆盖提供的每一个 beat_id，按原始顺序排列。每段镜头秒数之和必须等于该段 duration_s。优先每段一镜。保持创意、运镜及声音约束。
返回 JSON {"shots":[...]}。每镜含 beat_id,duration_sec(正整数),scene_id,characters(稳定角色ID数组),
visual_description,action,camera(含shot_size,movement,angle三个字符串),start_state,end_state,
transition(首镜opening，其余cut或continue),continuity_requirements(字符串数组),dialogue(仅实际对白)。
ID跨镜稳定。只在连续运镜并需要继承上一末帧时标 continue，否则 cut。连续镜开始状态直接复制上一末状态。
主动纠正源剧本和草稿中的空间矛盾：若人物需要以鞋尖相碰，将其开场站位设为同侧、脚边有小水洼、相隔半步且能自然触及。全部相关画面描述与前后状态都要一致。只对涉及该动作的剧本应用此例。检查动作能否在时段内完成，距离不可能时调整站位，保持故事意图。落实指定收尾运镜，不添加台词。
若附有ViMax草稿，使用其有价值的构图并服从原始剧本。"""

async def vimax_draft(chat,script):
    from agents.storyboard_artist import StoryboardArtist
    from langchain_core.runnables import RunnableLambda
    async def invoke(messages):
        return (await chat.ainvoke(messages)).content
    artist=StoryboardArtist(RunnableLambda(invoke))
    rows=await artist.design_storyboard(json.dumps(script['beats'],ensure_ascii=False),[],
        json.dumps(script['brief'],ensure_ascii=False))
    return [row.model_dump() for row in rows]

def build_storyboard(script,backend='auto',budget='standard',chat=None):
    if script.get('status')=='needs_input':
        return {'status':'needs_input','question':script.get('question'),'shots':[]}
    if script.get('status')!='ready':
        raise ValueError('需要 ready 剧本')
    beats=script['beats']
    ends={}
    cursor=0
    narration={}
    for b in beats:
        k=b['beat_id']
        d=b['duration_s']
        if k in ends or type(d)!=int or d<=0 or b['start_s']!=cursor or b['end_s']!=cursor+d:
            raise ValueError('剧本时间线或段落ID错误')
        cursor+=d
        ends[k]=d
        narration[k]=b.get('narration','')
    if not beats or cursor!=script['duration_s']:
        raise ValueError('剧本总时长错误')
    brief=script['brief']
    if (brief['duration_mode']=='max' and cursor>brief['seconds']) or (brief['duration_mode']=='exact' and cursor!=brief['seconds']):
        raise ValueError('剧本违反时长要求')
    if backend not in ('auto','vimax','baseline') or budget not in ('economy','standard','ample'):
        raise ValueError('无效配置')
    if backend=='auto':
        camera=brief.get('camera','').strip().lower()
        backend='vimax' if budget=='ample' and (camera not in ('','unspecified','未指定') or brief.get('story_complexity')=='high') else 'baseline'
    if chat is None:
        load_config()
        chat=Chat()
    payload={k:script[k] for k in ('brief','beats','title','source_idea')}
    if backend=='vimax':
        payload['draft']=asyncio.run(vimax_draft(chat,script))
    rows=chat.structured(PROMPT,payload)['shots']
    # Allocate each beat's locked integer seconds in proportion to proposed shot lengths.
    for key, available in ends.items():
        group = [r for r in rows if r.get('beat_id') == key]
        weights = [r.get('duration_sec') for r in group]
        if not group or len(group) > available or any(type(w)!=int or w<=0 for w in weights):
            raise ValueError(f'缺少段落或镜头过密: {key}')
        if sum(weights) != available:
            spare = available - len(group)
            shares = [divmod(spare*w, sum(weights)) for w in weights]
            lengths = [1 + q for q, _ in shares]
            for i in sorted(range(len(group)), key=lambda i: -shares[i][1])[:available-sum(lengths)]:
                lengths[i] += 1
            for r, length in zip(group, lengths):
                r['duration_sec'] = length
    shots=[]
    seen=[]
    totals=dict.fromkeys(ends,0)
    cursor=0
    for row in rows:
        k=row['beat_id']
        d=row['duration_sec']
        if k not in ends or type(d)!=int or d<=0:
            raise ValueError('无效分镜段落或时长')
        if not seen or seen[-1]!=k:
            seen.append(k)
        totals[k]+=d
        for field in ('scene_id','start_state','end_state','visual_description','action'):
            if not isinstance(row.get(field),str) or not row[field].strip():
                raise ValueError(field)
        for field in ('characters','continuity_requirements'):
            if not isinstance(row.get(field),list) or not all(isinstance(x,str) for x in row[field]):
                raise ValueError(field)
        if not all(isinstance(row['camera'].get(x),str) and row['camera'][x].strip() for x in ('movement','angle','shot_size')):
            raise ValueError('camera')
        trans=row['transition']
        if trans not in (('opening',) if not shots else ('cut','continue')):
            raise ValueError('transition')
        if trans=='continue' and row['start_state']!=shots[-1]['end_state']:
            raise ValueError('连续镜首末状态不匹配')
        if not isinstance(row.get('dialogue'),str) or (brief.get('sound')=='silent' and row['dialogue'].strip()):
            raise ValueError('dialogue')
        shots.append({**row,'shot_id':f'shot_{len(shots)+1:03}','order':len(shots)+1,'start_s':cursor,'end_s':cursor+d,
            'input_state_from':shots[-1]['shot_id'] if trans=='continue' else None})
        cursor+=d
    if seen!=list(ends) or totals!=ends:
        raise ValueError('分镜段落顺序或时长不匹配')
    return dict(status='ready',schema_version=1,title=script['title'],brief=brief,duration_s=cursor,
                narration_by_beat=narration,shots=shots,backend=backend)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('script',type=Path)
    p.add_argument('--backend',default='auto')
    p.add_argument('--budget',default='standard')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=build_storyboard(json.loads(a.script.read_text(encoding='utf-8')),a.backend,a.budget)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
