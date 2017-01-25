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
require 'LinearTransform'

torch.setheaptracking(true)

-- =========================================== COMMAND LINE OPTIONS ====================================================

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-max_batch_size', 50)
cmd:option('-max_seq_length', 50)
cmd:option('-load_bucketed_training_set', '/homes/iws/kingb12/data/BillionWords/50k_V_bucketed_set.th7')
cmd:option('-train_file', '')
cmd:option('-wmap_file', "/homes/iws/kingb12/data/BillionWords/50k_V_word_map.th7")
cmd:option('-wfreq_file', "/homes/iws/kingb12/data/BillionWords/50k_V_word_freq.th7")

-- Model options
cmd:option('-init_from', '')
cmd:option('-reset_iterations', 1)
cmd:option('-wordvec_size', 100)
cmd:option('-hidden_size', 100)
cmd:option('-dropout', 0)

-- Optimization options
cmd:option('-max_epochs', 50)
cmd:option('-learning_rate', 0.1)
cmd:option('-lr_decay', 0.3)

--Output Options
cmd:option('-print_example_loss', false)
cmd:option('-save_model_at_epoch', false)
cmd:option('-save_prefix', '/homes/iws/kingb12/LanguageModelRNN/newcudamodel')


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
-- load the training set
if opt.train_file == '' and opt.load_bucketed_training_set ~= nil then
    train_set = torch.load(opt.load_bucketed_training_set)
else
    local train_file = "/homes/iws/kingb12/data/BillionWords/50k_V_train_small.th7"
    local word_map_file = "/homes/iws/kingb12/data/BillionWords/50k_V_word_map.th7"
    local word_freq_file = "/homes/iws/kingb12/data/BillionWords/50k_V_word_freq.th7"
    train_set = bucket_training_set(torch.load(train_file))
end
train_set = clean_dataset(train_set, opt.max_batch_size, opt.max_seq_length)
function train_set:size()
    return #train_set
end

-- parameters and settings. Use cmd.opt soon --
local embeddingSize = opt.wordvec_size
local learningRate = opt.learning_reate
local learningRateDecay = opt.lr_decay
local max_epochs = opt.max_epochs
local dropout = opt.dropout > 0
local hiddenSize = opt.hidden_size
local vocabSize = 50000


-- =========================================== THE MODEL ===============================================================

-- The Word Embedding Layer --
-- word-embeddings can be learned using a LookupTable. Training is faster if they are supplied pre-trained, which can be done by changing
-- the weights at the index for a given word to its embedding form word2vec, etc. This is a doable next-step
local emb_layer = nn.LookupTable(vocabSize, embeddingSize)

lm = nn.Sequential()
-- Input Layer: Embedding LookupTable
lm:add(emb_layer) -- takes a sequence of word indexes and returns a sequence of word embeddings

-- Hidden Layers: Two LSTM layers, stacked
-- next steps: dropout, etc.
lm:add(nn.LSTM(embeddingSize, hiddenSize))
if dropout then lm:add(nn.Dropout(opt.dropout)) end
lm:add(nn.LSTM(hiddenSize, hiddenSize))
if dropout then lm:add(nn.Dropout(opt.dropout)) end
lm:add(nn.LSTM(hiddenSize, hiddenSize))
if dropout then lm:add(nn.Dropout(opt.dropout)) end
lm:add(nn.DynamicView(hiddenSize)) -- to transform to the appropriate dimmensions
lm:add(nn.Linear(hiddenSize, vocabSize))

-- Output Layer: LogSoftMax. Outputs are a distribution over each word in the vocabulary x seqLength*batchSize
lm:add(nn.LogSoftMax())

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
        torch.save(opt.save_prefix..sgd_trainer._epoch_Number..'.th7')
    end
    sgd_trainer._example_Number  = sgd_trainer._example_Number + 1
end

sgd_trainer.hookIteration = print_info