--
-- Created by IntelliJ IDEA.
-- User: bking
-- Date: 1/12/17
-- Time: 9:44 AM
-- To change this template use File | Settings | File Templates.
--

-- allows local imports
package.path = ';/Users/bking/IdeaProjects/LanguageModelRNN/?.lua;'..package.path

-- While unrelated to language modeling, this is a simple step 0 task to get aquainted with the nn package and torch

require 'nn'
require 'torch'
require 'util'

-- ============================================= PREPARING THE DATA ====================================================
-- This is a dataset consisting of a table of input, label pairs, where input is a torch.ByteTensor and label is a number class
dataset = dataset_from_csv('data/train.csv')

-- We received ByteTensors of shape (783 x 1) and we would like (28 x 28). To fix this, we need to resize and reshape
print('Beginning input tensor shape: ')
print(dataset[1][1]:size())

for i = 1, #dataset do
    local x = dataset[i][1]:resize(784):reshape(28, 28)
    dataset[i][1] = torch.DoubleTensor(1, 28, 28)
    dataset[i][1][1] = x:double()
    dataset[i][2] = torch.ByteTensor({dataset[i][2] + 1})
end

print('Current input tensor shape: ')
print(dataset[1][1]:size())

-- =============================================== SPECIFYING THE NN ARCHITECTURE ======================================


-- mlp is the wrapper for our NN. It is a sequential NN, in that layers are stacked operate as a simple feed-forward NN.
-- General gist of this architecture is that we are doing a single convolution and pooling, followed by a couple of fully
-- connected layers, and a final LogSoftMax for a probability over the 10 classes in {0-9}. ReLU is our chosen non-linearity
-- for convenience and quick training.
net = nn.Sequential()
-- a spatial convolution layer
net:add(nn.SpatialConvolution(1, 6, 5, 5))
net:add(nn.ReLU())
-- a max pooling layer (chooses max value from 2x2 squares, reduces information)
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.ReLU())
-- Re shape it to single dimmensional vectors, we no longer care about 2D structure
net:add(nn.View(864))
net:add(nn.Linear(864, 120))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())                       -- non-linearity
net:add(nn.Linear(120, 84))
net:add(nn.ReLU())                       -- non-linearity
net:add(nn.Linear(84, 10))               -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax()) -- final non-linearity puts makes classification a probabilitiy distribution

-- =============================================== SPECIFYING THE CRITERION ============================================

-- Here we use a negative log-likelihood criterion, which works well when our network outputs log-likelihoods (via LogSoftMax as the final layer)
criterion = nn.ClassNLLCriterion()

-- =============================================== TRAINING ============================================================

--  We train using Stochastic Gradient Descent. We set a learning rate and number of epochs of training
sgd_trainer = nn.StochasticGradient(net, criterion)
sgd_trainer.learningRate = 0.005
sgd_trainer.maxIteration = 5

-- now, sgd_trainer is prepared to do all the training for us, the reason we use the NN package in the first place instead of manipulating the tensors ourselves.
-- sgd_trainer:train(dataset)

-- =============================================== EVALUATION ==========================================================



