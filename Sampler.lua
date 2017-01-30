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
    self.argmax = true
end

function layer:updateOutput(input)
    if self.argmax then
        self.output = self:argMax(input)
    else
        self.output = self:sampleDistribution(input)
    end
    return self.output
end

function layer:sampleDistribution(input)
    local sample = torch.IntTensor(input:size()[1])
    for i=1,input:size()[1] do
        local sorted, indices = input[i]:sort()
        local sum = 0.0
        local value = math.random()
        for j=1,sorted:size()[1] do
            local index = sorted:size()[1] + 1 - j
            sum = sum + sorted[index]
            if sum > value then
                sample[i] = indices[index]
                break
            end
        end
    end
    return sample
end

function layer:argMax(input)
    local sample = torch.IntTensor(input:size()[1])
    for i=1,input:size()[1] do
        local max = input[i][1]
        local argmax = 1
        for j=1,input[i]:size()[1] do
            if input[i][j] > max then
                max = input[i][j]
                argmax = j
            end
        end
        sample[i] = argmax
    end
    return sample
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

