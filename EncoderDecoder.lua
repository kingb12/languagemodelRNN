--
-- Created by IntelliJ IDEA.
-- User: bking
-- Date: 2/8/17
-- Time: 10:53 AM
-- To change this template use File | Settings | File Templates.
--

package.path = ';/homes/iws/kingb12/LanguageModelRNN/?.lua;'..package.path

require 'torch'
require 'nn'
require 'nnx'
require 'nngraph'
require 'util'
-- require 'torch-rnn'
require 'DynamicView'
require 'Sampler'
require 'optim'
require 'LSTM'
require 'TemporalCrossEntropyCriterion'
require 'encdec_eval_functions'

-- =========================================== COMMAND LINE OPTIONS ====================================================

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-enc_inputs', '../data/rl_enc_inputs.th7')
cmd:option('-dec_inputs', '../data/rl_dec_inputs.th7')
cmd:option('-outputs', '../data/rl_outputs.th7')
cmd:option('-in_lengths', '../data/rl_in_lengths.th7')
cmd:option('-out_lengths', '../data/rl_out_lengths.th7')
cmd:option('-helper', '../data/rl_helper.th7')

cmd:option('-valid_enc_inputs', '/homes/iws/kingb12/data/rl_venc_inputs.th7')
cmd:option('-valid_dec_inputs', '/homes/iws/kingb12/data/rl_vdec_inputs.th7')
cmd:option('-valid_outputs', '/homes/iws/kingb12/data/rl_voutputs.th7')
cmd:option('-valid_in_lengths', '/homes/iws/kingb12/data/rl_vin_lengths.th7')
cmd:option('-valid_out_lengths', '/homes/iws/kingb12/data/rl_vout_lengths.th7')

cmd:option('-max_in_len', 200, 'max encoder sequence length')
cmd:option('-max_out_len', 300, 'max decoder sequence length')
cmd:option('-min_out_len', 1, 'min encoder sequence length')
cmd:option('-min_in_len', 1, 'min decoder sequence length')
cmd:option('-batch_size', 4)

-- Model options
cmd:option('-init_enc_from', '')
cmd:option('-init_dec_from', '')
cmd:option('-wordvec_size', 100)
cmd:option('-hidden_size', 512)
cmd:option('-dropout', 0)
cmd:option('-num_enc_layers', 1)
cmd:option('-num_dec_layers', 1)
cmd:option('-weights', '')
cmd:option('-no_average_loss', false)
cmd:option('-enc_remember_states', false)
cmd:option('-dec_remember_states', false)

-- Optimization options
cmd:option('-max_epochs', 50)
cmd:option('-learning_rate', 0.1)
cmd:option('-lr_decay', 0.0)
cmd:option('-algorithm', 'adam')

--Output Options
cmd:option('-print_loss_every', 1000)
cmd:option('-save_model_at_epoch', false)
cmd:option('-save_prefix', '/homes/iws/kingb12/LanguageModelRNN/')
cmd:option('-backup_save_dir', '')
cmd:option('-run', false)
cmd:option('-print_acc_every', 0)
cmd:option('-print_examples_every', 0, 'how often to print out samples')
cmd:option('-valid_loss_every', 0)


-- Backend options
cmd:option('-gpu', false)

local opt = cmd:parse(arg)
local tensorType = 'torch.FloatTensor'
local learningRate = opt.learning_rate
local dropout = opt.dropout > 0.0

-- Choosing between GPU/CPU Mode
if opt.gpu then
    tensorType = 'torch.CudaTensor'
    require 'cutorch'
    require 'cunn'
end

-- ============================================= DATA ==================================================================

-- loaded from saved torch files
enc_inputs = torch.load(opt.enc_inputs) -- title <begin> ingredients
dec_inputs = torch.load(opt.dec_inputs) -- recipe
outputs = torch.load(opt.outputs) -- recipe shifted one over (like a LM)
in_lengths = torch.load(opt.in_lengths) -- lengths specifying end of padding
out_lengths = torch.load(opt.out_lengths) -- lengths specifying end of padding
helper = torch.load(opt.helper) -- has word_map, reverse, etc.
local vocab_size = #helper.n_to_w
local wmap = helper.n_to_w

-- =========================================== THE MODEL ===============================================================

-- ***** ENCODER *****

local lu = nn.LookupTable(vocab_size, opt.wordvec_size)
local enc_lu, dec_lu = lu, lu:clone('weight', 'gradWeight')
if opt.init_enc_from == '' then
    -- The Word Embedding Layer --
    -- word-embeddings can be learned using a LookupTable. Training is faster if they are supplied pre-trained, which can be done by changing
    -- the weights at the index for a given word to its embedding form word2vec, etc. This is a doable next-step
    enc = nn.Sequential()
    -- Input Layer: Embedding LookupTable
    enc:add(enc_lu) -- takes a sequence of word indexes and returns a sequence of word embeddings
    enc._rnns = {}

    -- Hidden Layers: Two LSTM layers, stacked
    -- next steps: dropout, etc.
    for i=1,opt.num_enc_layers do
        local lstm
        if i == 1 then
            lstm = nn.LSTM(opt.wordvec_size, opt.hidden_size)
        else
            lstm = nn.LSTM(opt.hidden_size, opt.hidden_size)
        end
        if opt.enc_remember_states then
            lstm.remember_states = true
        end
        enc._rnns[#enc._rnns + 1] = lstm
        enc:add(lstm)
        if dropout then enc:add(nn.Dropout(opt.dropout)) end
    end
    
    
else
    -- load a model from a th7 file
    enc = torch.load(opt.init_enc_from)

end

-- ***** DECODER *****

if opt.init_dec_from == '' then
    -- Input Layer: Embedding LookupTable. Same as one for encoder so we don't learn two embeddings per word.
    -- we'll be building the decoder with nngraph so we can reuse the lookup layer and pass along hidden state from encoder in training,
    -- since h0 and c0 are graph inputs, we need to make a node for them, done with Identity() layer. nngraph overrides nn.Module()({graph parents})
    local dec_c0 = nn.Identity()()
    local dec_h0 = nn.Identity()()
    local dec_lu = dec_lu()
    dec_rnns = {}

    -- Hidden Layers: N LSTM layers, stacked, with optional dropout. previous helps us form a linear graph with these
    local previous, drop
    for i=1,opt.num_dec_layers do
        local lstm, lstm_n
        if i == 1 then
            lstm = nn.LSTM(opt.wordvec_size, opt.hidden_size)
            lstm_n = lstm({dec_c0, dec_h0, dec_lu})
            previous = lstm_n
        else
            lstm = nn.LSTM(opt.hidden_size, opt.hidden_size)
            lstm_n = lstm(previous)
            previous = lstm_n
        end
        if opt.dec_remember_states then
            lstm.remember_states = true
        end
        dec_rnns[#dec_rnns + 1] = lstm
        if opt.dropout > 0.0 then 
            drop = nn.Dropout(opt.dropout)(previous)
            print("ading dropout to dec")
            previous = drop
        end
    end
    -- now linear transition layers
    local dec_v1 = nn.View(-1, opt.hidden_size)(previous)
    print(previous == drop)
    local dec_lin = nn.Linear(opt.hidden_size, vocab_size)(dec_v1)

    -- now combine them into a graph module
    dec = nn.gModule({dec_c0, dec_h0, dec_lu}, {dec_lin}) -- {inputs}, {outputs}
    dec._rnns = dec_rnns

else
    -- load a model from a th7 file
    dec = torch.load(opt.init_dec_from)
end

-- =============================================== TRAINING ============================================================

-- Training --
-- We'll use TemporalCrossEntropyCriterion to maximize the likelihood for correct words, ignoring 0 which indicates padding.

criterion = nn.TemporalCrossEntropyCriterion()
local cb = torch.CudaTensor.zeros(torch.CudaTensor.new(), opt.batch_size, opt.hidden_size)
local hzeros = torch.CudaTensor.zeros(torch.CudaTensor.new(), opt.batch_size, opt.max_in_len-1, opt.hidden_size)

-- logging
if opt.save_model_at_epoch then
    logger = optim.Logger(opt.save_prefix .. '.log')
    logger:setNames{'Epoch','Training Loss.', 'Learning Rate:  ', 'T. Perplexity. ', 'V. Loss', 'V. Perplexity'}
    logger:display(false) -- prevents display on remote hosts
    logger:style{'+-'} -- points and lines for plot
end

if opt.gpu then
    criterion = criterion:cuda()
    enc = enc:cuda()
    dec = dec:cuda()
end

local params, gradParams = combine_all_parameters(enc, dec)
local batch = 1
local epoch = 0
local embs
local loss_this_epoch = 0
local perp_this_epoch = 0
local v_loss, v_perp

local function print_info(learningRate, iteration, currentError, v_loss, v_perp, t_perp)
    print("Current Iteration: ", iteration)
    print("Current Loss: ", currentError)
    print("Current Learing Rate: ", learningRate)
    if opt.save_model_at_epoch then
        pcall(torch.save, opt.save_prefix..'_enc.th7', enc)
        pcall(torch.save, opt.save_prefix..'_dec.th7', dec)
        logger:add{epoch, currentError, learningRate, t_perp, v_loss, v_perp}
        logger:plot()
        if (opt.backup_save_dir ~= '') then 
            pcall(torch.save, opt.backup_save_dir..opt.save_prefix..'_enc.th7', enc)
            pcall(torch.save, opt.backup_save_dir..opt.save_prefix..'_dec.th7', dec)
        end    
    end
end

local optim_config = {learningRate = learningRate }


local function feval(params)
    gradParams:zero()

    -- retrieve inputs for this batch
    local enc_input = enc_inputs[batch]
    local dec_input = dec_inputs[batch]
    local output = outputs[batch]

    -- forward pass
    for _,v in pairs(enc._rnns) do v:resetStates() end
    for _,v in pairs(dec._rnns) do v:resetStates() end
    local enc_fwd = enc:forward(enc_input) -- enc_fwd is h1...hN
    local dec_h0 = enc_fwd[{{}, opt.max_in_len, {}}] -- grab the last hidden state from the encoder, which will be at index max_in_len
    local dec_fwd = dec:forward({cb:clone(), dec_h0, dec_input}) -- forwarding a new zeroed cell state, the encoder hidden state, and frame-shifted expected output (like LM)
    dec_fwd = torch.reshape(dec_fwd, opt.batch_size, opt.max_out_len, vocab_size)
    local loss = criterion:forward(dec_fwd, output) -- loss is essentially same as if we were a language model, ignoring padding
    _, embs = torch.max(dec_fwd, 3)
    embs = torch.reshape(embs, opt.batch_size, opt.max_out_len)

    -- backward pass
    local cgrd = criterion:backward(dec_fwd, output)
    cgrd = torch.reshape(cgrd, opt.batch_size*opt.max_out_len, vocab_size)
    local hlgrad, dgrd = table.unpack(dec:backward({dec_h0, dec_input}, cgrd))
    local hlgrad = torch.reshape(hlgrad, opt.batch_size, 1, opt.hidden_size)
    local hgrad = torch.cat(hzeros, hlgrad, 2)
    local egrd = enc:backward(enc_input, hgrad)

    --update batch/epoch
    if batch == enc_inputs:size(1) then
        batch = 1
        epoch = epoch + 1
    else
        batch = batch + 1
    end

    return loss, gradParams
end

function train_model()
    while (epoch < opt.max_epochs) do
        local examples = (batch-1)*opt.batch_size
        local output = outputs[batch]
        local out_length = out_lengths[batch]
        local in_length = in_lengths[{{examples+1, examples+opt.batch_size}}]
        local _, loss = run_one_batch(opt.algorithm)
        local normed_loss = loss[1] / (torch.sum(out_length) / enc_inputs[batch]:size(1))
        loss_this_epoch = loss_this_epoch + (normed_loss / enc_inputs:size(1))
        perp_this_epoch = perp_this_epoch + (torch.exp(normed_loss) / enc_inputs:size(1))

        if (batch % opt.print_loss_every) == 0 then print('Loss: ', loss_this_epoch) end

        -- print info
        if (batch == 1) then
            if (epoch % opt.valid_loss_every == 0) then
                v_loss, v_perp  = get_validation_loss(valid_enc_inputs, valid_dec_inputs, valid_outputs, valid_in_lengths, valid_out_lengths)
            end
            print_info(optim_config.learningRate, epoch, loss_this_epoch, v_loss, v_perp, perp_this_epoch)
            loss_this_epoch = 0.0
            perp_this_epoch = 0.0
        end

        -- print accuracy (handled here so we don't have to pass dec_fwd/embs out of feval)
        if batch % opt.print_acc_every == 0 then
            local acc, nwords = 0, 0
            for n = 1, opt.batch_size do
                nwords = nwords + out_length[n]
                for t = 1, out_length[n] do
                    if embs[n][t] == output[n][t] then
                        acc = acc + 1
                    end
                end
            end
            acc = acc / nwords
            print('Accuracy: ', acc)
        end

        -- print examples
        if batch % opt.print_examples_every == 0 then
            local enc_input = enc_inputs[batch]
            local dec_input = dec_inputs[batch]
            local output = outputs[batch]
            local closs = 0
            for i = 1, opt.batch_size do
                io.write('Encoder Input: ')
                for j = 1, opt.max_in_len do
                    io.write(wmap[enc_input[i][j]] .. ' ')
                end
                print('')
                io.write('Decoder Input: ')
                for j =1, opt.max_out_len do
                    io.write(wmap[dec_input[i][j]] .. ' ')
                end
                print('')
                io.write('Decoder Output: ')
                for j = 1, opt.max_out_len do
                    io.write(wmap[embs[i][j]] .. ' ')
                end
                print('')
                io.write('Ground Truth: ')
                for j = 1, opt.max_out_len do
                    io.write(wmap[output[i][j]] .. ' ')
                end
                print('')
                print('***********')
            end
            print('------------------')
        end
    end
end

function run_one_batch(algorithm)
    if algorithm == 'adam' then
        return optim.adam(feval, params, optim_config)
    else
        return optim.sgd(feval, params, optim_config)
    end
end

function get_validation_loss(venc_inputs, vdec_inputs, voutputs, vin_lengths, vout_lengths)
    enc:evaluate()
    dec:evaluate()
    local v_perp, v_loss = perplexity_over_dataset(enc, dec, venc_inputs, vdec_inputs, vin_lengths, vout_lengths, voutputs)
    return v_loss, v_perp
end

if (opt.valid_loss_every > 0) then
    valid_enc_inputs = torch.load(opt.valid_enc_inputs)
    valid_dec_inputs = torch.load(opt.valid_dec_inputs)
    valid_outputs = torch.load(opt.valid_outputs)
    valid_in_lengths = torch.load(opt.valid_in_lengths)
    valid_out_lengths = torch.load(opt.valid_out_lengths)
    v_loss, v_perp = get_validation_loss(valid_enc_inputs, valid_dec_inputs, valid_outputs, valid_in_lengths, valid_out_lengths)
end


if opt.run then
    train_model()
end

