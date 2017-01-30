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

torch.setheaptracking(true)

-- =========================================== COMMAND LINE OPTIONS ====================================================

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-max_batch_size', 50)
cmd:option('-max_seq_length', 50)
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
cmd:option('-dropout', 0)
cmd:option('-num_layers', 3)

-- Optimization options
cmd:option('-max_epochs', 50)
cmd:option('-learning_rate', 0.1)
cmd:option('-lr_decay', 0.3)

--Output Options
cmd:option('-print_example_loss', false)
cmd:option('-save_model_at_epoch', false)
cmd:option('-save_prefix', '/homes/iws/kingb12/LanguageModelRNN/newcudamodel')
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
local vocabSize = 50000


-- =========================================== THE MODEL ===============================================================

if opt.init_from ~= '' then
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
        lm:add(nn.LSTM(embeddingSize, hiddenSize))
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

-- Training --
-- We'll use NLLCriterion to maximize the likelihood for correct words, and StochasticGradientDescent to run it.
criterion = nn.ClassNLLCriterion()

--  We train using Stochastic Gradient Descent. We set a learning rate and number of epochs of training
if opt.gpu then
    sgd_trainer = nn.StochasticGradient(lm:cuda(), criterion:cuda())
else
    sgd_trainer = nn.StochasticGradient(lm, crtierion)
end
sgd_trainer.learningRate = learningRate
sgd_trainer.learningRateDecay = learningRateDecay
sgd_trainer.maxIteration = max_epochs
sgd_trainer._epoch_Number = 1

local function print_info(self, iteration, currentError)
    print("Current Iteration: ", iteration)
    print("Current Loss: ", currentError)
    print("Current Learing Rate: ", self.learningRate)
    if opt.save_model_at_epoch then
        torch.save(opt.save_prefix..'.th7', self.module)
    end
    sgd_trainer._epoch_Number  = sgd_trainer._epoch_Number + 1
end

sgd_trainer.hookIteration = print_info

if opt.run then
    sgd_trainer:train(train_set)
end

-- =============================================== SAMPLING ============================================================

sampler = nn.Sequential()
sampler:add(nn.Exp())
sampler:add(nn.Sampler())

