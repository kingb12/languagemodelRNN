from Bleu.calculatebleu import *
import sys,json

def closest_bleu_match(candidates, references):
    result = []
    for i,c in enumerate(candidates):
        best_score = 0
        best_match = ''
        score = 0
#        print "Candidate: ", i
        for j,r in enumerate(references):
 #           print j
            try:
                score = BLEU([c], [[r]])
            except ZeroDivisionError:
                continue
            if score >= best_score:
                best_score = score
                best_match = r
            if score == 1.0:
                break
        result.append({'candidate': c, 'best_score': 100*best_score, 'best_match': best_match})
    return result

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        cands = json.loads(f.read())
    with open(sys.argv[2], 'r') as f:
        refs = json.loads(f.read())
    print json.dumps(closest_bleu_match(cands, refs))
