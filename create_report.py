import json, sys

with open('EncDecReportTemplate.ipynb', 'r') as f:
   template = json.loads(f.read())

with open(sys.argv[1], 'w') as f:
    template['cells'][2]['source'][0] = u'report_file = \'' + unicode(sys.argv[2]) + u'\'\n'
    template['cells'][2]['source'][1] = u'log_file = \'' + unicode(sys.argv[3]) + u'\'\n'
    f.write(json.dumps(template))
