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
if opt.gpu then 
    require 'cutorch'
    require 'cunn'
end

function clean_dataset(dataset, batch_size)
    new_set = {}
    for i=1, #dataset do
        if dataset[i][1]:size(1) == batch_size then
            new_set[#new_set + 1] = dataset[i]
        end
    end
    return new_set
end
train_set = clean_dataset(torch.load(opt.train_set), 50)
valid_set = clean_dataset(torch.load(opt.valid_set), 50)
test_set = clean_dataset(torch.load(opt.test_set), 50)
model = torch.load(opt.model)
criterion = nn.ClassNLLCriterion()

function table_cuda(dataset) 
    for i=1, #dataset do
        dataset[i][1] = dataset[i][1]:cuda()
        dataset[i][2] = dataset[i][2]:cuda()
    end
    return dataset
end


-- CUDA everything if GPU
if opt.gpu then
    train_set = table_cuda(train_set)
    valid_set = table_cuda(valid_set)
    test_set = table_cuda(test_set)
    model = model:cuda()
    criterion = criterion:cuda()
end
function loss_on_dataset(data_set, criterion)
    local loss = 0.0
    local batch_losses = {}
    for i=1, #data_set do
        local l, n = criterion:forward(model:forward(data_set[i][1]), data_set[i][2])
        if batch_losses[data_set[i][1]:size(2)] == nil then
            batch_losses[data_set[i][1]:size(2)] = {l}
        else 
            local x = batch_losses[data_set[i][1]:size(2)]
            x[#x + 1] = l
        end
        loss = loss + (l / n)
    end
    return loss, batch_losses
end

-- We will build a report as a table which will be converted to json.
output = {}

-- calculate losses
print('Calculating Training Loss...')
local tr_set_loss, tr_batch_loss = loss_on_dataset(train_set, criterion)
output['train_loss'] = tr_set_loss
output['train_batch_loss'] = tr_batch_loss

print('Calculating Validation Loss...')
local vd_set_loss, vd_batch_loss = loss_on_dataset(valid_set, criterion)
output['valid_loss'] = vd_set_loss
output['valid_batch_loss'] = vd_batch_loss

print('Calculating Test Loss...')
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

