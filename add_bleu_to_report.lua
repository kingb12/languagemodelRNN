cjson = require 'cjson'

require 'encdec_eval_functions'

torch.setheaptracking(true)

-- =========================================== COMMAND LINE OPTIONS ====================================================

local cmd = torch.CmdLine()
-- Options
cmd:option('-add_bleu_to', '')
cmd:option('-out', '')
local opt = cmd:parse(arg)

if opt.add_bleu_to ~= '' then
    local f = io.open(opt.add_bleu_to, 'r')
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
    output['bleu'] = calculate_bleu(references, candidates)
    local f = io.open(opt.out, 'w+')
    f:write(cjson.encode(output))
    f:close()
end
