cjson = require 'cjson'

require 'encdec_eval_functions'

torch.setheaptracking(true)

-- =========================================== COMMAND LINE OPTIONS ====================================================

local cmd = torch.CmdLine()
-- Options

cmd:option('-add_to', '')
cmd:option('-out', '')
cmd:option('-bleu', false)
cmd:option('-n_pairs_bleu', false)
cmd:option('-avg_alignment', false)
local opt = cmd:parse(arg)
if opt.add_to ~= '' then
    local f = io.open(opt.add_to, 'r')
    output = cjson.decode(f:read())
    f:close()
    local references = {}
    local candidates = {}
    for i=1,#output['train_samples'] do
        candidates[#candidates + 1] = output['train_samples'][i]['generated']
        references[#references + 1] = output['train_samples'][i]['gold']
    end
    for i=1,#output['valid_samples'] do
        candidates[#candidates + 1] = output['valid_samples'][i]['generated']
        references[#references + 1] = output['valid_samples'][i]['gold']
    end
    for i=1,#output['test_samples'] do
        candidates[#candidates + 1] = output['test_samples'][i]['generated']
        references[#references + 1] = output['test_samples'][i]['gold']
    end
    if opt.bleu then
        output['bleu'] = calculate_bleu(references, candidates)
    end
    if opt.n_pairs_bleu then
        output['n_pairs_bleu_generated'] = n_pairs_bleu(candidates, 1000)
        output['n_pairs_bleu_gold'] = n_pairs_bleu(references, 1000)
    end
    if opt.avg_alignment then
        output['average_alignment_generated'] = alignment_scores(candidates)
        output['average_alignment_gold'] = alignment_scores(references)
    end
    local f = io.open(opt.out, 'w+')
    f:write(cjson.encode(output))
    f:close()
end
