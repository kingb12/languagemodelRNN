--
-- Created by IntelliJ IDEA.
-- User: bking
-- Date: 2/19/17
-- Time: 12:10 PM
-- To change this template use File | Settings | File Templates.
--

package.path = ';/homes/iws/kingb12/LanguageModelRNN/?.lua;'..package.path

require 'torch'
require 'nn'
require 'nnx'
require 'nngraph'
require 'util'
require 'DynamicView'
require 'TemporalCrossEntropyCriterion'
require 'LSTM'
require 'Sampler'
cjson = require 'cjson'
require 'cutorch'
require 'cunn'
require 'encdec_eval_functions'

torch.setheaptracking(true)

-- =========================================== COMMAND LINE OPTIONS ====================================================

local cmd = torch.CmdLine()
-- Options
cmd:option('-calculate_perplexity', false)
cmd:option('-generate_samples', false)


-- Dataset options
cmd:option('-train_enc_inputs', '/homes/iws/kingb12/data/rl_enc_inputs.th7')
cmd:option('-train_dec_inputs', '/homes/iws/kingb12/data/rl_dec_inputs.th7')
cmd:option('-train_outputs', '/homes/iws/kingb12/data/rl_outputs.th7')
cmd:option('-train_in_lengths', '/homes/iws/kingb12/data/rl_in_lengths.th7')
cmd:option('-train_out_lengths', '/homes/iws/kingb12/data/rl_out_lengths.th7')

cmd:option('-valid_enc_inputs', '/homes/iws/kingb12/data/rl_venc_inputs.th7')
cmd:option('-valid_dec_inputs', '/homes/iws/kingb12/data/rl_vdec_inputs.th7')
cmd:option('-valid_outputs', '/homes/iws/kingb12/data/rl_voutputs.th7')
cmd:option('-valid_in_lengths', '/homes/iws/kingb12/data/rl_vin_lengths.th7')
cmd:option('-valid_out_lengths', '/homes/iws/kingb12/data/rl_vout_lengths.th7')

cmd:option('-test_enc_inputs', '/homes/iws/kingb12/data/rl_tenc_inputs.th7')
cmd:option('-test_dec_inputs', '/homes/iws/kingb12/data/rl_tdec_inputs.th7')
cmd:option('-test_outputs', '/homes/iws/kingb12/data/rl_toutputs.th7')
cmd:option('-test_in_lengths', '/homes/iws/kingb12/data/rl_tin_lengths.th7')
cmd:option('-test_out_lengths', '/homes/iws/kingb12/data/rl_tout_lengths.th7')

cmd:option('-helper', '../data/rl_helper.th7')

-- Model options
cmd:option('-enc', 'enc.th7')
cmd:option('-dec', 'dec.th7')

--Output Options
cmd:option('-num_samples', 10)
cmd:option('-max_sample_length', 10)
cmd:option('-max_gen_example_length', 10)
cmd:option('-no_arg_max', false)
cmd:option('-out', '')

local opt = cmd:parse(arg)
-- ================================================ EVALUATION =========================================================

train_enc_inputs = torch.load(opt.train_enc_inputs)
train_dec_inputs = torch.load(opt.train_dec_inputs)
train_outputs = torch.load(opt.train_outputs)
train_in_lengths = torch.load(opt.train_in_lengths)
train_out_lengths = torch.load(opt.train_out_lengths)

valid_enc_inputs = torch.load(opt.valid_enc_inputs)
valid_dec_inputs = torch.load(opt.valid_dec_inputs)
valid_outputs = torch.load(opt.valid_outputs)
valid_in_lengths = torch.load(opt.valid_in_lengths)
valid_out_lengths = torch.load(opt.valid_out_lengths)

test_enc_inputs = torch.load(opt.test_enc_inputs)
test_dec_inputs = torch.load(opt.test_dec_inputs)
test_outputs = torch.load(opt.test_outputs)
test_in_lengths = torch.load(opt.test_in_lengths)
test_out_lengths = torch.load(opt.test_out_lengths)

enc = torch.load(opt.enc)
dec = torch.load(opt.dec)
helper = torch.load(opt.helper)

enc:evaluate()
dec:evaluate()

for k,v in pairs(enc._rnns) do v.remember_states = false end
for k,v in pairs(dec._rnns) do v.remember_states = false end

criterion = nn.TemporalCrossEntropyCriterion():cuda()

-- We will build a report as a table which will be converted to json.
output = {}


sampler =  nn.Sampler():cuda()
if opt.no_arg_max then
    sampler.argmax = false
end

function sample(encoder, decoder, enc_state, sequence, max_samples)
    if max_samples == nil then
        max_samples = 1
    end
    if enc_state == nil then
        enc_state = encoder:forward(sequence)
        sequence = torch.CudaTensor({helper.w_to_n['<beg>']}):reshape(1, 1)
    end
    local cb = torch.CudaTensor.zeros(torch.CudaTensor.new(), 1, enc_state:size(3))
    local addition = torch.zeros(sequence:size(1)):cuda()
    local output = torch.cat(sequence, addition , 2)
    local dec_h0 = enc_state[{{}, enc_state:size(2), {}}] -- grab the last hidden state from the encoder, which will be at index max_in_len
    print(dec_h0:size(), enc_state:size(),cb:size())
    local y = decoder:forward({cb:clone(), dec_h0, sequence})
    local sampled = sampler:forward(y)
    for i=1, output:size(1) do output[i][output:size(2)] = sampled[output:size(2) - 1] end
    if max_samples == 1 or helper.n_to_w[output[1][output:size(2)]] == '</S>' then
        return output
    else
        return sample(encoder, decoder, enc_state, output, max_samples - 1)
    end
end

function sequence_to_string(seq)
    local str = ''
    if seq:dim() == 2 then seq = seq[1] end
    for i=1, seq:size()[1] do
        local next_word = helper.n_to_w[seq[i]] == nil and '<UNK2>' or helper.n_to_w[seq[i]]
        str = str..' '..next_word
    end
    return str
end

function generate_samples(data_set, outputs, num_samples)
    local results = {}
    print('Generating Samples...')
    for i = 1, num_samples do
        print('Sample ', i)
        local t_set_idx = (torch.random() % data_set:size(1)) + 1
        if t_set_idx > data_set:size(1) then t_set_idx = 1 end
        local example = data_set[t_set_idx]
        local example_no = torch.random() % example:size(1) + 1
        if example_no > example:size(1) then example_no = 1 end
        local x = example[example_no]
        x = x:reshape(1, x:size(1))
        local result = {}
        result['generated'] = sequence_to_string(sample(enc, dec, nil, x, opt.max_sample_length))
        result['gold'] = sequence_to_string(outputs[t_set_idx][example_no])
        results[#results + 1] = result
    end
    return results
end

local cb = torch.CudaTensor.zeros(torch.CudaTensor.new(), train_enc_inputs[1]:size(1), enc:forward(train_enc_inputs[1]):size(3))

-- calculate perplexity
function perplexity_over_dataset(enc, dec, enc_inputs, dec_inputs, in_lengths, out_lengths, outputs)
    local data_perplexity = 0
    for i=1,enc_inputs:size(1) do
        for _,v in pairs(enc._rnns) do v:resetStates() end
        for _,v in pairs(dec._rnns) do v:resetStates() end
        local enc_input = enc_inputs[i]
        local dec_input = dec_inputs[i]
        local output = outputs[i]
        local enc_fwd = enc:forward(enc_input) -- enc_fwd is h1...hN
        local dec_h0 = enc_fwd[{{}, enc_inputs[1]:size(2), {}}] -- grab the last hidden state from the encoder, which will be at index max_in_len
        local dec_fwd = dec:forward({cb:clone(), dec_h0, dec_input}) -- forwarding a new zeroed cell state, the encoder hidden state, and frame-shifted expected output (like LM)
        dec_fwd = torch.reshape(dec_fwd, enc_input:size(1), dec_input:size(2), #helper.n_to_w)
        local loss = criterion:forward(dec_fwd, output) -- loss is essentially same as if we were a language model, ignoring padding
        loss = loss / (torch.sum(out_lengths[i])
        local batch_perplexity = torch.exp(loss)
        data_perplexity = data_perplexity + (batch_perplexity / enc_inputs:size(1))
    end
    return data_perplexity
end

if opt.calculate_perplexity then
    print('Calculating Training Perplexity...')
    local tr_perp = perplexity_over_dataset(model, train_set)
    output['train_perplexity'] = tr_perp
    print('Calculating Valid Perplexity...')
    local vd_perp = perplexity_over_dataset(model, valid_set)
    output['valid_perplexity'] = vd_perp
    print('Calculating Test Perplexity...')
    local ts_perp = perplexity_over_dataset(model, test_set)
    output['test_perplexity'] = ts_perp
end

-- generate some samples
if opt.generate_samples then
    output['train_samples'] = generate_samples(train_set, opt.num_samples)
    output['valid_samples'] = generate_samples(valid_set, opt.num_samples)
    output['test_samples'] = generate_samples(test_set, opt.num_samples)
end

output['architecture'] = {}
output['architecture']['encoder'] = tostring(enc)
output['architecture']['decoder'] = tostring(dec)

if opt.out ~= '' then
    local s = cjson.encode(output)
    local io = require 'io'
    local f = io.open(opt.out, 'w+')
    f:write(s)
    f:close()
end
