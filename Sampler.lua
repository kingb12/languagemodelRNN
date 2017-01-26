--
-- Created by IntelliJ IDEA.
-- User: bking
-- Date: 1/26/17
-- Time: 2:00 PM
-- To change this template use File | Settings | File Templates.
--

require 'torch'
require 'nn'
require 'math'

-- This class represents a sampling layer. Ite expects vectors representing a distribition (MUST sum to 1)

local layer, parent = torch.class('nn.Sampler', 'nn.Module')

function layer:__init()
    parent.__init(self)
end

function layer:updateOutput(input)
    self.output = torch.IntTensor(input:size()[1])
    for i=1,input:size()[1] do
        local sorted, indices = input[i]:sort()
        local sum = 0.0
        local value = math.random()
        for j=1,sorted:size()[1] do
            local index = sorted:size() + 1 - j
            sum = sum + sorted[index]
            if sum > value then
                self.output[i] = indices[index]
                break
            end
        end
    end
    return self.output
end

function layer:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput
    return self.gradInput
end

function layer:accGradParameters(input, gradOutput)
    return gradOutput
end

function layer:updateParamaters()
    return nil
end

function layer:parameters()
    return nil
end


function layer:training()
    parent.training(self)
end

function layer:evaluate()
    parent.evaluate(self)
end

