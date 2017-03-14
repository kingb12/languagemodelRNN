import json, sys, os

with open(sys.argv[2], 'r') as f:
   template = json.loads(f.read())

with open(sys.argv[1], 'w') as f:
    report_files = []
    log_files = []
    for i in range(3, len(sys.argv)):
        report_files.append(unicode(os.getcwd()) + u'/' + unicode(sys.argv[i]) + u'/' + unicode(sys.argv[i]) + u'.json')
        log_files.append(unicode(os.getcwd()) + u'/' + unicode(sys.argv[i]) + u'/' + unicode(sys.argv[i]) + u'_logs.json')
    
    template['cells'][2]['source'][0] = u'report_files = ' + json.dumps(report_files) +  u'\n'
    template['cells'][2]['source'][1] = u'log_files = ' + json.dumps(log_files) + u'\n'
    f.write(json.dumps(template))
