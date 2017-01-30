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
cmd:option('-calculate_losses', false)
cmd:option('-generate_samples', false)


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
wmap = torch.load(opt.wmap_file)

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
        loss = loss + l
    end
    loss = loss / #data_set
    return loss, batch_losses
end

-- We will build a report as a table which will be converted to json.
output = {}

-- calculate losses
if opt.calculate_losses then
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
end

sampler = opt.gpu and nn.Sampler():cuda() or nn.Sampler()

function sample(model, sequence, max_samples)
    if max_samples == nil then
        max_samples = 1
    end
    local addition = opt.gpu and torch.zeros(sequence:size(1)):cuda() or torch.zeros(sequence:size(1))
    local output = torch.cat(sequence, addition , 2)
    local y = model:forward(sequence:repeatTensor(50, 1))
    local sampled = sampler:forward(y:reshape(50, y:size(1) / 50, y:size(2))[1])
    for i=1, output:size(1) do output[i][output:size(2)] = sampled[output:size(2) - 1] end
    if max_samples == 1 or wmap[output[i][output:size(2)]] == '</S>' then
        return output
    else
        return sample(output, max_samples - 1)
    end
end

function sequence_to_string(seq)
    local str = ''
    if seq:dim() == 2 then seq = seq[1] end
    for i=1, seq:size()[1] do
        local next_word = wmap[seq[i]] == nil and '<UNK>' or wmap[seq[i]]
        str = str..' '..next_word
    end
    return str
end

function generate_samples(data_set, num_samples)
    local results = {}
    for i = 1, num_samples do
        local t_set_idx = (torch.random() % #data_set) + 1
        local example = data_set[t_set_idx][1]
        local label = data_set[t_set_idx][2]
        local example_no = torch.random() % example:size(1)
        local cut_length = (torch.random() % example:size(2)) + 1
        local x = opt.gpu and torch.CudaTensor(1, cut_length) or torch.IntTensor(1, cut_length)
        for i=1, cut_length do x[1][i] = example[example_no][i] end
        local result = {}
        result['generated'] = sequence_to_string(sample(model, x, opt.max_sample_length))
        result['gold'] = sequence_to_string(label:reshape(example:size())[example_no])
        result['supplied_length'] = cut_length
        results[#results + 1] = result
    end
    return results
end

-- generate some samples
if opt.generate_samples then

end



