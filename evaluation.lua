--
-- Created by IntelliJ IDEA.
-- User: bking
-- Date: 1/30/17
-- Time: 2:25 AM
-- To change this template use File | Settings | File Templates.
--

package.path = ';/homes/iws/kingb12/LanguageModelRNN/?.lua;'..package.path

require 'torch'
require 'nn'
require 'nnx'
require 'util'
require 'torch-rnn'
require 'DynamicView'
require 'Sampler'
cjson = require 'cjson'

torch.setheaptracking(true)

-- =========================================== COMMAND LINE OPTIONS ====================================================

local cmd = torch.CmdLine()
-- Options
cmd:option('-gpu', false)

-- Dataset options
cmd:option('-train_set', '/homes/iws/kingb12/data/BillionWords/25k_V_bucketed_set.th7')
cmd:option('-valid_set', '/homes/iws/kingb12/data/BillionWords/25k_V_bucketed_valid_set.th7')
cmd:option('-test_set', '/homes/iws/kingb12/data/BillionWords/25k_V_bucketed_test_set.th7')
cmd:option('-wmap_file', "/homes/iws/kingb12/data/BillionWords/25k_V_word_map.th7")
cmd:option('-wfreq_file', "/homes/iws/kingb12/data/BillionWords/25k_V_word_freq.th7")

-- Model options
cmd:option('-model', 'newcudamodel.th7')

--Output Options
cmd:option('-batch_loss_file', '')
cmd:option('-num_samples', 10)
cmd:option('-max_sample_length', 10)
cmd:option('-run', false)

local opt = cmd:parse(arg)
-- ================================================ EVALUATION =========================================================

train_set = torch.load(opt.train_set)
valid_set = torch.load(opt.valid_set)
test_set = torch.load(opt.test_set)
model = torch.load(opt.model)
criterion = nn.ClassNLLCriterion()

-- CUDA everything if GPU
if opt.gpu then
    require 'cutorch'
    require 'cunn'
    train_set = train_set:cuda()
    valid_set = valid_set:cuda()
    test_set = test_set:cuda()
    model = model:cuda()
    criterion = criterion:cuda()
end

function loss_on_dataset(data_set, criterion)
    local loss = 0.0
    local batch_losses = {}
    for i=1, data_set:size() do
        local l, n = criterion:forward(model:forward(data_set[i][1]), data_set[i][2])
        batch_losses[i] = l
        loss = loss + (l / n)
    end
    return loss, batch_losses
end

-- We will build a report as a table which will be converted to json.
output = {}

-- calculate losses
local tr_set_loss, tr_batch_loss = loss_on_dataset(train_set, criterion)
output['train_loss'] = tr_set_loss
output['train_batch_loss'] = tr_batch_loss

local vd_set_loss, vd_batch_loss = loss_on_dataset(valid_set, criterion)
output['valid_loss'] = vd_set_loss
output['valid_batch_loss'] = vd_batch_loss

local ts_set_loss, ts_batch_loss = loss_on_dataset(test_set, criterion)
output['test_loss'] = ts_set_loss
output['test_batch_loss'] = ts_batch_loss

sampler = nn.Sampler()
function sample(output, max_samples)
    if max_samples == nil then
        max_samples = 1
    end
    local output = torch.cat(output, torch.zeros(output:size(1)))
    local sampled = sampler:forward(output)
    for i=1, output:size(1) do output[i][output:size(2) + 1] = sampled[i] end
    if max_samples == 1 then
        return output
    else
        return sample(output, max_samples - 1)
    end
end

-- generate some samples
for i=1, opt.num_samples do

end



