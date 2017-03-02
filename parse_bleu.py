import sys, json

result = {}
s = (''.join(line for line in sys.stdin if line.startswith('BLEU'))).split(' ')
result['score'] = float(s[2].split(',')[0])
result['components'] = dict([(i, float(s[3].split('/')[i-1])) for i in range(1,5)])
print json.dumps(result)
