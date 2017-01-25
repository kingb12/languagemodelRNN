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

-- parameters and settings. Use cmd.opt soon --
embeddingSize = 100
accUpdate = false
pretrain = true
pretrained_file = ""
learningRate = 0.5
hiddenSize = 100
train_file = "/homes/iws/kingb12/data/BillionWords/50k_V_train_small.th7"
word_map_file = "/homes/iws/kingb12/data/BillionWords/50k_V_word_map.th7"
word_freq_file = "/homes/iws/kingb12/data/BillionWords/50k_V_word_freq.th7"
inputSize = embeddingSize
seq_length = 50
batch_size = 5
vocabSize = 50000


-- Loading the Dataset --

ds = torch.load(train_file) -- sequence data as integer indexes to wmap
wmap = torch.load(word_map_file) -- map index -> "word"
wfreq = torch.load(word_freq_file) -- relative word frequencies. Used for SoftMaxTree

-- Shaping the Data --

-- The data comes in a long sequence of words. Here we'll shape it into batches of batchSize sequences of length seqLength
-- where each training example is a sequence of seqLength words.

print('bucketing')
train_set = torch.load('/homes/iws/kingb12/data/BillionWords/50k_V_bucketed_set.th7')
function train_set:size()
    return #train_set
end
-- The Word Embedding Layer --

-- word-embeddings can be learned using a LookupTable. Training is faster if they are supplied pre-trained, which can be done by changing
-- the weights at the index for a given word to its embedding form word2vec, etc. This is a doable next-step
local emb_layer = nn.LookupTable(#wmap, embeddingSize)


-- The Model --

lm = nn.Sequential()
-- Input Layer: Embedding LookupTable
lm:add(emb_layer) -- takes a sequence of word indexes and returns a sequence of word embeddings

-- Hidden Layers: Two LSTM layers, stacked
-- next steps: dropout, etc.
lm:add(nn.LSTM(embeddingSize, hiddenSize))
lm:add(nn.LSTM(hiddenSize, hiddenSize))
lm:add(nn.LSTM(hiddenSize, hiddenSize))
lm:add(nn.DynamicView(hiddenSize)) -- to transform to the appropriate dimmensions
lm:add(nn.Linear(hiddenSize, vocabSize))



-- Output Layer: LogSoftMax. Outputs are a distribution over each word in the vocabulary x seqLength*batchSize
lm:add(nn.LogSoftMax())



-- Training --
-- We'll use NLLCriterion to maximize the likelihood for correct words, and StochasticGradientDescent to run it.
criterion = nn.ClassNLLCriterion()
-- =============================================== TRAINING ============================================================

--  We train using Stochastic Gradient Descent. We set a learning rate and number of epochs of training
sgd_trainer = nn.StochasticGradient(lm, criterion)
sgd_trainer.learningRate = learningRate
sgd_trainer.learningRateDecay = 0.5
sgd_trainer.maxIteration = 25
sgd_trainer._example_Number = 1
local function print_info(self, iteration, currentError)
    print("Current Iteration: ", iteration)
    print("Current Loss: ", currentError)
    print("Current Learing Rate: ", self.learningRate)
    torch.save('/homes/iws/kingb12/LanguageModelRNN/model'..sgd_trainer._example_Number..'.th7')
    sgd_trainer._example_Number  = sgd_trainer._example_Number + 1
end

sgd_trainer.hookIteration = print_info






