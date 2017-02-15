--
-- Created by IntelliJ IDEA.
-- User: bking
-- Date: 1/12/17
-- Time: 9:11 AM
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

torch.setheaptracking(true)

-- =========================================== COMMAND LINE OPTIONS ====================================================

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-max_batch_size', 50)
cmd:option('-max_seq_length', 30)
cmd:option('-no_dataset', false)
cmd:option('-load_bucketed_training_set', '/homes/iws/kingb12/data/BillionWords/25k_V_bucketed_set.th7')
cmd:option('-train_file', '')
cmd:option('-wmap_file', "/homes/iws/kingb12/data/BillionWords/25k_V_word_map.th7")
cmd:option('-wfreq_file', "/homes/iws/kingb12/data/BillionWords/25k_V_word_freq.th7")

-- Model options
cmd:option('-init_from', '')
cmd:option('-reset_iterations', 1)
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
cmd:option('-algorithm', 'sgd')

--Output Options
cmd:option('-print_loss_every', 1000)
cmd:option('-save_model_at_epoch', false)
cmd:option('-save_prefix', '/homes/iws/kingb12/LanguageModelRNN/newcudamodel')
cmd:option('-save_prefix_backup', '')
cmd:option('-run', false)


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

if not opt.no_dataset then
    -- load the training set
    if opt.train_file == '' and opt.load_bucketed_training_set ~= nil then
        train_set = torch.load(opt.load_bucketed_training_set)
    else
        local train_file = opt.train_file
        local word_map_file = opt.wmap_file
        local word_freq_file = opt.wfreq_file
        train_set = bucket_training_set(torch.load(train_file))
    end
    train_set = clean_dataset(train_set, opt.max_batch_size, opt.max_seq_length, tensorType)
    local t_set = {}
    -- Must use equal batch sizes in order to remember states
    for i=1,#train_set do
        if train_set[i][1]:size(1) == 50 and train_set[i][1]:size(2) <= opt.max_seq_length then
            t_set[#t_set + 1] = train_set[i]
        end
    end
    train_set = t_set
    function train_set:size()
        return #train_set
    end
end

local embeddingSize = opt.wordvec_size
local learningRate = opt.learning_rate
local learningRateDecay = opt.lr_decay
local max_epochs = opt.max_epochs
local dropout = opt.dropout > 0
local hiddenSize = opt.hidden_size
local vocabSize = opt.vocab_size


-- =========================================== THE MODEL ===============================================================

if opt.init_from == '' then
    -- The Word Embedding Layer --
    -- word-embeddings can be learned using a LookupTable. Training is faster if they are supplied pre-trained, which can be done by changing
    -- the weights at the index for a given word to its embedding form word2vec, etc. This is a doable next-step
    local emb_layer = nn.LookupTable(vocabSize, embeddingSize)

    lm = nn.Sequential()
    -- Input Layer: Embedding LookupTable
    lm:add(emb_layer) -- takes a sequence of word indexes and returns a sequence of word embeddings

    -- Hidden Layers: Two LSTM layers, stacked
    -- next steps: dropout, etc.
    for i=1,opt.num_layers do
        local lstm
        if i == 1 then
            lstm = nn.LSTM(embeddingSize, hiddenSize)
        else
            lstm = nn.LSTM(hiddenSize, hiddenSize)
        end
        lstm.remember_states = true
        lm:add(lstm)
        if dropout then lm:add(nn.Dropout(opt.dropout)) end
    end

    lm:add(nn.DynamicView(hiddenSize)) -- to transform to the appropriate dimmensions
    lm:add(nn.Linear(hiddenSize, vocabSize))

    -- Output Layer: LogSoftMax. Outputs are a distribution over each word in the vocabulary x seqLength*batchSize
    lm:add(nn.LogSoftMax())
else
    -- load a model from a th7 file
    lm = torch.load(opt.init_from)
end


-- =============================================== TRAINING ============================================================
-- logging
if opt.save_model_at_epoch then
    logger = optim.Logger(opt.save_prefix .. '.log')
    logger:setNames{'Epoch','Training Loss.', 'Learning Rate:  '}
    logger:display(false) -- prevents display on remote hosts
    logger:style{'+-'} -- points and lines for plot
end
-- Training --
-- We'll use NLLCriterion to maximize the likelihood for correct words, and StochasticGradientDescent to run it.
if opt.weights == 'inverse_freq' then
    local wfreq = torch.load(opt.wfreq_file)
    local inverse_wfreq = torch.pow(wfreq:double(), -1)
    criterion = nn.ClassNLLCriterion(inverse_wfreq)
elseif opt.weights == 'inverse_logfreq' then
    local wfreq = torch.load(opt.wfreq_file)
    local inverse_logwfreq = torch.pow(torch.log(wfreq:double()), -1)
    criterion = nn.ClassNLLCriterion(inverse_logwfreq)
else
    criterion = nn.ClassNLLCriterion()
end
if opt.no_avg_loss then
    criterion.sizeAverage = false
end
if opt.gpu then
    criterion = criterion:cuda()
    lm = lm:cuda()
end
local params, gradParams = combine_all_parameters(lm)
local batch = 1
local epoch = 0
local loss_this_epoch = 0.0

local function print_info(learningRate, iteration, currentError)
    print("Current Iteration: ", iteration)
    print("Current Loss: ", currentError)
    print("Current Learing Rate: ", learningRate)
    if opt.save_model_at_epoch then
        pcall(torch.save, opt.save_prefix..'.th7', lm)
        logger:add{epoch - 1, currentError, learningRate}
        logger:plot()

    end
    if opt.save_prefix_backup ~= '' then
        pcall(torch.save, opt.save_prefix_backup..opt.save_prefix..'.th7', lm)
    end
end

optim_config = {learningRate = learningRate }

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
            loss_this_epoch = loss_this_epoch + (loss[1] / #train_set)
            if (batch % opt.print_loss_every) == 0 then print('Loss: ', loss_this_epoch) end
            if (batch == 1) then
                print_info(optim_config.learningRate, epoch, loss_this_epoch)
                loss_this_epoch = 0.0
            end
        end
    else
        while (epoch < max_epochs) do
            local _, loss = optim.sgd(feval, params, optim_config)
            loss_this_epoch = loss_this_epoch + (loss[1] / #train_set)
            if (batch % opt.print_loss_every) == 0 then print('Loss: ', loss_this_epoch) end
            if (batch == 1) then
                print_info(optim_config.learningRate, epoch, loss_this_epoch)
                loss_this_epoch = 0.0
            end
        end
    end
end

if opt.run then
    train_model()
end

-- =============================================== SAMPLING ============================================================

sampler = nn.Sequential()
sampler:add(nn.Exp())
sampler:add(nn.Sampler())

-- =============================================== EVALUATION ==========================================================
