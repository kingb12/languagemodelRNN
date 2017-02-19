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
cmd:option('-calculate_perplexity', false)
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
cmd:option('-max_gen_example_length', 10)
cmd:option('-no_arg_max', false)
cmd:option('-out', '')

local opt = cmd:parse(arg)
-- ================================================ EVALUATION =========================================================
if opt.gpu then 
    require 'cutorch'
    require 'cunn'
end

function xclean_dataset(dataset, batch_size, max_seq_length)
    local new_set = {}
    for i=1, #dataset do
        if dataset[i][1]:size(1) == batch_size and dataset[i][1]:size(2) <= max_seq_length then
            new_set[#new_set + 1] = dataset[i]
        end
    end
    return new_set
end
train_set = xclean_dataset(torch.load(opt.train_set), 50, 30)
valid_set = xclean_dataset(torch.load(opt.valid_set), 50, 30)
test_set = xclean_dataset(torch.load(opt.test_set), 50, 30)
model = torch.load(opt.model)
model:get(2).remember_states = false
model:get(4).remember_states = false
model:evaluate()
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
if opt.no_arg_max then 
    sampler.argmax = false
end

function sample(model, sequence, max_samples)
    if max_samples == nil then
        max_samples = 1
    end
    local addition = opt.gpu and torch.zeros(sequence:size(1)):cuda() or torch.zeros(sequence:size(1))
    local output = torch.cat(sequence, addition , 2)
    local y = model:forward(sequence)
    local sampled = sampler:forward(y)
    for i=1, output:size(1) do output[i][output:size(2)] = sampled[output:size(2) - 1] end
    if max_samples == 1 or wmap[output[1][output:size(2)]] == '</S>' then
        return output
    else
        return sample(model, output, max_samples - 1)
    end
end

function sequence_to_string(seq)
    local str = ''
    if seq:dim() == 2 then seq = seq[1] end
    for i=1, seq:size()[1] do
        local next_word = wmap[seq[i]] == nil and '<UNK2>' or wmap[seq[i]]
        str = str..' '..next_word
    end
    return str
end

function truncate_dataset(data_set, max_seq_len)
    local result = {}
    for i=1, #data_set do
        if data_set[i][1]:size(2) <= max_seq_len then
            result[#result + 1] = data_set[i]
        end
    end
    return result
end

function generate_samples(data_set, num_samples)
    local results = {}
    if opt.max_gen_example_length > 0 then
        data_set = truncate_dataset(data_set, opt.max_gen_example_length)
    end
    print('Generating Samples...')
    for i = 1, num_samples do
        print('Sample ', i)
        local t_set_idx = (torch.random() % #data_set) + 1
        if t_set_idx > #data_set then t_set_idx = 1 end
        local example = data_set[t_set_idx][1]
        local label = data_set[t_set_idx][2]
        local example_no = torch.random() % example:size(1) + 1
        if example_no > example:size(1) then example_no = 1 end
        local cut_length = (torch.random() % example:size(2)) + 1
        if cut_length > example:size(2) then cut_length = 1 end
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

-- calculate perplexity
function perplexity_over_dataset(model, data_set)
    local data_perplexity = 0
    local batch_perps = {}
    for i=1,#data_set do
        local y = model:forward(data_set[i][1])
        local loss = criterion:forward(y, data_set[i][2])
        local batch_perplexity = torch.exp(loss)
            if batch_perps[data_set[i][1]:size(2)] == nil then
                batch_perps[data_set[i][1]:size(2)] = {batch_perplexity}
            else
                local x = batch_perps[data_set[i][1]:size(2)]
                x[#x + 1] = batch_perplexity
            end
        data_perplexity = data_perplexity + (batch_perplexity / #data_set)
    end
    return data_perplexity, batch_perps
end

if opt.calculate_perplexity then
    print('Calculating Training Perplexity...')
    local tr_perp, tr_bps = perplexity_over_dataset(model, train_set)
    output['train_perplexity'] = tr_perp
    output['train_batch_perplexities'] = tr_bps
    print('Calculating Valid Perplexity...')
    local vd_perp, vd_bps = perplexity_over_dataset(model, valid_set)
    output['valid_perplexity'] = vd_perp
    output['valid_batch_perplexities'] = vd_bps
    print('Calculating Test Perplexity...')
    local ts_perp, ts_bps = perplexity_over_dataset(model, test_set)
    output['test_perplexity'] = ts_perp
    output['test_batch_perplexities'] = ts_bps
end

-- generate some samples
if opt.generate_samples then
    output['train_samples'] = generate_samples(train_set, opt.num_samples)
    output['valid_samples'] = generate_samples(valid_set, opt.num_samples)
    output['test_samples'] = generate_samples(test_set, opt.num_samples)
end

if opt.out ~= '' then
    local s = cjson.encode(output)
    local io = require 'io'
    local f = io.open(opt.out, 'w+')
    f:write(s)
    f:close()
end
