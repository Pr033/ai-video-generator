"""Compare the existing writer and the official ViMax Screenwriter."""
import asyncio
import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from script_agent import generate, BRIEF, WRITE

class Chat:
    def __init__(self):
        self.calls = 0
        self.tokens = 0

    def request(self, messages):
        self.calls += 1
        body = dict(model=os.environ['CHAT_MODEL'], messages=messages, temperature=0.5)
        req = urllib.request.Request(os.environ['CHAT_BASE_URL'].rstrip('/')+'/chat/completions',
            data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+os.environ['CHAT_API_KEY'],'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=90) as response:
            result=json.load(response)
        self.tokens += result.get('usage',{}).get('total_tokens',0)
        return result['choices'][0]['message']['content']

    async def ainvoke(self,messages):
        converted=[{'role':'user' if role=='human' else role,'content':content} for role,content in messages]
        return SimpleNamespace(content=await asyncio.to_thread(self.request,converted))

    def structured(self,instruction,data):
        raw=self.request([{'role':'system','content':instruction},{'role':'user','content':json.dumps(data,ensure_ascii=False)}]).strip()
        if raw.startswith('```'):
            raw='\n'.join(raw.splitlines()[1:-1])
        return json.loads(raw)

def write_vimax(data,chat):
    from agents.screenwriter import Screenwriter
    writer=Screenwriter(chat)
    requirements=json.dumps({'brief':data['brief'],'slots_seconds':data['slots_seconds']},ensure_ascii=False)
    async def draft():
        story=await writer.develop_story(data['idea'],requirements)
        scenes=await writer.write_script_based_on_story(story,requirements)
        return story,scenes
    story,scenes=asyncio.run(draft())
    output=chat.structured(WRITE+' 将以下ViMax场景映射到指定时段。保留人物、事件和顺序。',{**data,'scenes':scenes})
    return output,{'story':story,'scenes':scenes}

def compare(idea,folder,slots=None,chat=None):
    chat=chat or Chat()
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    def save(name,obj):
        (folder/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    brief=chat.structured(BRIEF,{'idea':idea})
    save('brief.json',brief)
    report={}
    for backend in ('baseline','vimax'):
        start=time.monotonic()
        calls,tokens=chat.calls,chat.tokens
        def invoke(instruction,data):
            if instruction==BRIEF:
                return brief
            if backend=='baseline':
                return chat.structured(instruction,data)
            output,raw=write_vimax(data,chat)
            save('vimax_raw.json',raw)
            return output
        try:
            script=generate(idea,invoke,slots)
            save(backend+'.json',script)
            status=script['status']
        except Exception as exc:
            status='error: '+str(exc)
        report[backend]=dict(status=status,calls=chat.calls-calls,tokens=chat.tokens-tokens,seconds=round(time.monotonic()-start,2))
        save('comparison.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('idea')
    p.add_argument('--out',required=True)
    p.add_argument('--slots',nargs='+',type=int)
    args=p.parse_args()
    print(compare(args.idea,args.out,args.slots))
