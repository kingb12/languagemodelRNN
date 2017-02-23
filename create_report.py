import json, sys, os

with open(sys.argv[2], 'r') as f:
   template = json.loads(f.read())

with open(sys.argv[1], 'w') as f:
    template['cells'][2]['source'][0] = u'report_file = \'' + unicode(os.getcwd()) + u'/' + unicode(sys.argv[3]) + u'\'\n'
    template['cells'][2]['source'][1] = u'log_file = \'' + unicode(os.getcwd()) + u'/' + unicode(sys.argv[4]) + u'\'\n'
    f.write(json.dumps(template))
