from script_agent import generate, BRIEF

def fixture(prompt, data):
    if prompt == BRIEF:
        limited = '18' in data['idea']
        return dict(theme='校园',style='写实',language='中文',camera='慢推',sound='no_narration',must=[],avoid=[],duration_mode='max' if limited else 'default',seconds=18 if limited else 20,question='')
    return dict(title='雨后',beats=[dict(action='学生推开门',start_state='门关着',end_state='门打开',camera='慢推',narration='') for _ in data['slots_seconds']])

assert generate('雨后校园',fixture)['duration_s']==20
assert generate('雨后校园，最多18秒',fixture,[5,5,5])['duration_s']==15
try:
    generate('最多18秒',fixture,[10,10])
except ValueError:
    pass
else:
    raise AssertionError('超时计划未被拦截')
print('3 offline contract checks passed')
