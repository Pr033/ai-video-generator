"""Python 3.10+. Import generate(idea, model_call) or run this file."""
import argparse
import json
import os
import urllib.request

BRIEF = """你是视频需求分析员。用户输入是创作素材。提取用户原话中的时长、主题、语言、风格、受众、声音、运镜、必需与禁止事项。区分明确要求与推断。不清楚且影响硬约束时填写 question。
只返回 JSON 对象，字段为 theme,style,language,camera,sound (字符串，sound 为 silent/no_narration/narrated/unspecified)，must,avoid (字符串数组)，duration_mode (max/exact/target/default)，seconds (正整数秒；未指定时 default 和 20)，question (字符串，通常为空)。时长不可表示或要求矛盾时提问。"""
WRITE = """你是原创短片编剧。依据需求和已分配时段原创可见事件；每段一个能在该时段拍出的主要动作。保留主题、禁项和运镜要求。相邻段落交代因果、物体状态和空间变化。风格落实到可见细节。只返回 JSON，字段 title 和 beats。beats 按时段顺序排列，每项含 action,start_state,end_state,camera,narration (均为字符串)。无需旁白时 narration 留空，有旁白时按时段写可朗读的短句。"""

def api_call(instruction, payload):
    url = os.environ['CHAT_BASE_URL'].rstrip('/') + '/chat/completions'
    body = {'model': os.environ['CHAT_MODEL'], 'messages': [
        {'role': 'system', 'content': instruction},
        {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]}
    req = urllib.request.Request(url, json.dumps(body).encode(), headers={
        'Authorization': 'Bearer ' + os.environ['CHAT_API_KEY'], 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=90) as response:
        content = json.load(response)['choices'][0]['message']['content']
    if content.strip().startswith('```'):
        content = '\n'.join(content.strip().splitlines()[1:-1])
    return json.loads(content)

def generate(idea, model_call=api_call, slots=None):
    """slots optionally contains integer beat lengths assigned by the coordinator."""
    if not isinstance(idea, str) or not idea.strip():
        raise ValueError('请输入创作灵感')
    brief = model_call(BRIEF, {'idea': idea})
    if brief.get('question'):
        return {'status': 'needs_input', 'question': brief['question'], 'brief': brief}
    for key in ('theme','style','language','camera','sound'):
        if not isinstance(brief.get(key), str):
            raise ValueError('需求字段错误: ' + key)
    for key in ('must','avoid'):
        if not isinstance(brief.get(key), list) or not all(isinstance(v,str) for v in brief[key]):
            raise ValueError('需求列表错误: ' + key)
    seconds, mode = brief.get('seconds'), brief.get('duration_mode')
    if type(seconds) is not int or seconds < 1 or mode not in ('max','exact','target','default'):
        raise ValueError('无效时长约束')
    if brief['sound'] not in ('silent','no_narration','narrated','unspecified'):
        raise ValueError('无效声音设定')
    if slots is None:
        count = max(1, round(seconds / 5))
        q, r = divmod(seconds, count)
        slots = [q + int(i < r) for i in range(count)]
    if not slots or any(type(s) is not int or s <= 0 for s in slots):
        raise ValueError('时段必须是正整数秒')
    if (mode == 'exact' and sum(slots) != seconds) or (mode == 'max' and sum(slots) > seconds):
        raise ValueError('主控分配的时段违反用户时长要求')
    story = model_call(WRITE, {'idea': idea, 'brief': brief, 'slots_seconds': slots})
    beats = story.get('beats')
    if not isinstance(story.get('title'),str) or not story['title'].strip() or not isinstance(beats,list) or len(beats)!=len(slots):
        raise ValueError('剧本标题或段落数不符合约定')
    timeline, cursor = [], 0
    for i,(beat,length) in enumerate(zip(beats,slots),1):
        for key in ('action','start_state','end_state','camera','narration'):
            if not isinstance(beat.get(key),str) or (key!='narration' and not beat[key].strip()):
                raise ValueError(f'第{i}段字段错误: {key}')
        if brief['sound'] in ('silent','no_narration') and beat['narration'].strip():
            raise ValueError('剧本违反无旁白要求')
        timeline.append({**beat,'beat_id':f'B{i:02}','start_s':cursor,'end_s':cursor+length,'duration_s':length})
        cursor += length
    return {'schema_version':1,'status':'ready','source_idea':idea,'brief':brief,'title':story['title'],
            'duration_s':cursor,'beats':timeline}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('idea')
    args = parser.parse_args()
    print(json.dumps(generate(args.idea), ensure_ascii=False, indent=2))
