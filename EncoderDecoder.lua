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

local params, gradParams = enc:getParameters()
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
    local outputs = lm:forward(train_set[batch][1])
    local loss = criterion:forward(outputs, train_set[batch][2])
    local dloss_doutputs = criterion:backward(outputs, train_set[batch][2])
    lm:backward(train_set[batch][1], dloss_doutputs)
    if batch == train_set:size() then
        batch = 1
        epoch = epoch + 1
    else
        batch = batch + 1
    end
    return loss, gradParams
end

-- sgd_trainer.learningRateDecay = learningRateDecay

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
        end
    end
end

if opt.run then
    train_model()
end

