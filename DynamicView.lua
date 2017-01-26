--
-- Created by IntelliJ IDEA.
-- User: bking
-- Date: 1/23/17
-- Time: 10:50 AM
-- To change this template use File | Settings | File Templates.
--

require 'torch'
require 'nn'

-- This class represents a linear layer that can be placed on top of LSTM layers that perform dropout, and will resize the incoming View
-- to the appropriate size according to incoming batch sizes and sequence lengths (allowing for bucketting cleanly). Vlocabulary size
-- must be supplied as an argument, and output can be plugged into a LogSoftMax layer.

local layer, parent = torch.class('nn.DynamicView', 'nn.Module')

function layer:__init(outputSize)
    parent.__init(self)
    self.outputSize = outputSize
    self.vocab_size = vocab_size
    self.view = nn.View(1, 1, -1):setNumInputDims(3)
end

function layer:updateOutput(input)
    local N, T = input:size(1), input:size(2)
    self.view:resetSize(N * T, self.outputSize)
    self.output = self.view:updateOutput(input)
    return self.output
end

function layer:updateGradInput(input, gradOutput)
    self.gradInput = self.view:updateGradInput(input, gradOutput)
    return self.gradInput
end

function layer:accGradParameters(input, gradOutput)
return self.view:accGradParameters(input, gradOutput)
end

function layer:updateParamaters()
    return self.view:updateParameters()
end

function layer:parameters()
    return self.view:parameters()
end


function layer:training()
    self.view:training()
    parent.training(self)
end

function layer:evaluate()
    self.view:evaluate()
    parent.evaluate(self)
end