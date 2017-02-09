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
require 'util'
require 'torch-rnn'
require 'DynamicView'
require 'Sampler'
require 'optim'
require 'TemporalCrossEntropyCriterion'

-- =========================================== COMMAND LINE OPTIONS ====================================================

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-enc_inputs', '../data/rl_enc_inputs.th7')
cmd:option('-dec_inputs', '../data/rl_dec_inputs.th7')
cmd:option('-outputs', '../data/rl_outputs.th7')
cmd:option('-in_lengths', '../data/rl_in_lengths.th7')
cmd:option('-out_lengths', '../data/rl_out_lengths.th7')

cmd:option('-max_in_len', 200, 'max encoder sequence length')
cmd:option('-max_out_len', 300, 'max decoder sequence length')
cmd:option('-min_out_len', 1, 'min encoder sequence length')
cmd:option('-min_in_len', 1, 'min decoder sequence length')
cmd:option('-batch_size', 4)

-- Model options
cmd:option('-init_enc_from', '')
cmd:option('-wordvec_size', 100)
cmd:option('-hidden_size', 100)
cmd:option('-vocab_size', 25000)
cmd:option('-dropout', 0)
cmd:option('-num_layers', 3)
cmd:option('-weights', '')
cmd:option('-no_average_loss', false)

-- Optimization options
cmd:option('-max_epochs', 50)
cmd:option('-learning_rate', 0.1)
cmd:option('-lr_decay', 0.0)
cmd:option('-algorithm', 'adam')

--Output Options
cmd:option('-print_loss_every', 1000)
cmd:option('-save_model_at_epoch', false)
cmd:option('-save_prefix', '/homes/iws/kingb12/LanguageModelRNN/')
cmd:option('-run', false)
cmd:option('-print_acc_every', 0)

-- Backend options
cmd:option('-gpu', false)

local opt = cmd:parse(arg)
local tensorType = 'torch.FloatTensor'

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

-- =========================================== THE MODEL ===============================================================

-- ***** ENCODER *****

if opt.init_enc_from == '' then
    -- The Word Embedding Layer --
    -- word-embeddings can be learned using a LookupTable. Training is faster if they are supplied pre-trained, which can be done by changing
    -- the weights at the index for a given word to its embedding form word2vec, etc. This is a doable next-step
    emb_layer = nn.LookupTable(vocabSize, opt.embedding_size)

    enc = nn.Sequential()
    -- Input Layer: Embedding LookupTable
    enc:add(emb_layer) -- takes a sequence of word indexes and returns a sequence of word embeddings
    enc_rnns = {}

    -- Hidden Layers: Two LSTM layers, stacked
    -- next steps: dropout, etc.
    for i=1,opt.num_enc_layers do
        local lstm
        if i == 1 then
            lstm = nn.LSTM(opt.embedding_size, opt.hidden_size)
        else
            lstm = nn.LSTM(opt.hidden_size, opt.hidden_size)
        end
        lstm.remember_states = true
        enc_rnns[#enc_rnns + 1] = lstm
        enc:add(lstm)
        if dropout then lm:add(nn.Dropout(opt.dropout)) end
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
    local dec_emb_layer = emb_layer:clone('weight', 'gradWeight')()
    local dec_c0 = nn.Identity()()
    local dec_h0 = nn.Identity()()
    dec_rnns = {}

    -- Hidden Layers: N LSTM layers, stacked, with optional dropout. previous helps us form a linear graph with these
    local previous
    for i=1,opt.num_dec_layers do
        local lstm
        if i == 1 then
            lstm = nn.LSTM(opt.embedding_size, opt.hidden_size)({dec_c0, dec_h0, dec_emb_layer})
            previous = lstm
        else
            lstm = nn.LSTM(opt.hidden_size, opt.hidden_size)(previous)
            previous = lstm
        end
        lstm.remember_states = true
        dec_rnns[#dec_rnns + 1] = lstm
        enc:add(lstm)
        if opt.dropout > 0.0 then 
            local drop = nn.Dropout(opt.dropout)(previous)
            previous = drop
        end
    end
    -- now linear transition layers
    local dec_v1 = nn.View(-1, opt.hidden_size)(previous)
    local dec_lin = nn.Linear(opt.hidden_size, vocab_size)(dec_v1)

    -- now combine them into a graph module
    dec = nn.gModule({dec_c0, dec_h0, dec_emb_layer}, {dec_lin}) -- {inputs}, {outputs}

else
    -- load a model from a th7 file
    dec = torch.load(opt.init_dec_from)
end

-- =============================================== TRAINING ============================================================

-- Training --
-- We'll use TemporalCrossEntropyCriterion to maximize the likelihood for correct words, ignoring 0 which indicates padding.

criterion = nn.TemporalCrossEntropyCriterion()

if opt.gpu then
    criterion = criterion:cuda()
    enc = enc:cuda()
    dec = dec:cuda()
end

local params, gradParams = combine_all_parameters(enc, dec)
local batch = 1
local epoch = 0

local function print_info(learningRate, iteration, currentError)
    print("Current Iteration: ", iteration)
    print("Current Loss: ", currentError)
    print("Current Learing Rate: ", learningRate)
    if opt.save_model_at_epoch then
        torch.save(opt.save_prefix..'.th7', lm)
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
    for _,v in pairs(enc_rnns) do v:resetStates() end
    for _,v in pairs(dec_rnns) do v:resetStates() end
    local enc_fwd = enc:forward(enc_input)
    local dec_h0 = enc_fwd[{{}, max_in_len, {}}]
    local dec_fwd = dec:forward({cb:clone(), dec_h0, dec_input})
    dec_fwd = torch.reshape(dec_fwd, batch_size, max_out_len, vocab_size)
    local loss = criterion:forward(dec_fwd, output)
    local _, embs = torch.max(dec_fwd, 3)
    embs = torch.reshape(embs, batch_size, max_out_len)

    -- backward pass
    local cgrd = criterion:backward(dec_fwd, output)
    cgrd = torch.reshape(cgrd, batch_size*max_out_len, vocab_size)
    local hlgrad, dgrd = table.unpack(dec:backward({dec_h0, dec_input}, cgrd))
    local hlgrad = torch.reshape(hlgrad, batch_size, 1, hidden_size)
    local hgrad = torch.cat(hzeros, hlgrad, 2)
    local egrd = enc:backward(enc_input, hgrad)

    --update batch/epoch
    if batch == enc_inputs:size(1) then
        batch = 1
        epoch = epoch + 1
    else
        batch = batch + 1
    end

    -- print accuracy
    if batch % opt.print_acc_every == 0 then
        local _, embs = torch.max(dec_fwd, 3)
        embs = torch.reshape(embs, batch_size, max_out_len)
        print('Accuracy: ', accuracy(embs))
    end

    return loss, gradParams
end

function train_model()
    if opt.algorithm == 'adam' then
        while (epoch < max_epochs) do
            local _, loss = optim.adam(feval, params, optim_config)
            if (batch % opt.print_loss_every) == 0 then print('Loss: ', loss[1]) end
            if (batch == 1) then
                print_info(optim_config.learningRate, epoch, loss[1])
            end
        end
    else
        while (epoch < max_epochs) do
            local _, loss = optim.sgd(feval, params, optim_config)
            if (batch % opt.print_loss_every) == 0 then print('Loss: ', loss[1]) end
            if (batch == 1) then
                print_info(optim_config.learningRate, epoch, loss[1])
            end
        end
    end
end

function accuracy(embs)
    local acc, nwords = 0, 0
    for n = 1, batch_size do
        nwords = nwords + out_length[n]
        for t = 1, out_length[n] do
            if embs[n][t] == output[n][t] then
                acc = acc + 1
            end
        end
    end
    acc = acc / nwords
end

if opt.run then
    train_model()
end

