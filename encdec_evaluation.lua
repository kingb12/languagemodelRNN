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
