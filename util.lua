-- This is a file containing self-written utilities for relevant tasks in Lua as an exercise
require 'torch'
require 'io'
require 'math'

-- function for parsing a csv file into a table of tables. Skips the header line. Does not handle input with commas.
function parse_csv(csv_file_name)
    local lines = {}
    local i = 1
    local fh = io.open(csv_file_name)
    local line = fh.read(fh) -- why double fh? ignore first line read
    while true do
        local l = {}
        line = fh.read(fh)
        if not line then break end -- EOF corresponds to a nil returned from read()
        local j = 1
        for token in string.gmatch(line, "([^,]+),%s*") do
            l[j] = token
            j = j + 1
        end
        lines[i] = l
        i = i + 1
    end
    return lines
end

-- returns a table of input, label pairs from a CSV file. Label is first element
function dataset_from_csv(csv_file_name)
    local csv = parse_csv(csv_file_name)
    local dataset = {}
    for i = 1, #csv do
        local label = csv[i][1]
        local input = torch.ByteTensor(783)
        for j = 2, #csv[i] do input[j - 1] = csv[i][j] end
        dataset[i] = {input, label}
    end
    function dataset:size() return #dataset end
    return dataset
end

-- given a file containing word embeddings, insert them as weights to nn.LookupTable module.
-- embeddings must be sorted same as indexes
function set_pretrained_enbeddings(embedding_file_name, lookup_layer)
    local i = 1
    for line in io.lines(embedding_file_name) do
        local vals = line:splitAtCommas()
        lookup_layer.weight[i] = torch.Tensor(vals) -- set the pretrained values in the matrix
        i = i + 1
    end
end

-- used when using a larger vocab size, requiring a SoftMaxTree layer as opposed to LogSoftMax Not needed here?
function frequencyTree(word_frequency, binSize)
    binSize = binSize or 100
    local wf = word_frequency
    local vals, indices = wf:sort()
    local tree = {}
    local id = indices:size(1)
    function recursiveTree(indices)
        if indices:size(1) < binSize then
            id = id + 1
            tree[id] = indices
            return
        end
        local parents = {}
        for start=1,indices:size(1),binSize do
            local stop = math.min(indices:size(1), start+binSize-1)
            local bin = indices:narrow(1, start, stop-start+1)
            assert(bin:size(1) <= binSize)
            id = id + 1
            table.insert(parents, id)
            tree[id] = bin
        end
        recursiveTree(indices.new(parents))
    end
    recursiveTree(indices)
    return tree, id
end

-- takes a dataset and reduces vocab size, substituting <UNK> for infrequent words, and adjusting the word_map,
-- word_freq as necessary
function reduce_vocab_size(dataset, word_map, word_frequency, new_size)
    local ds = dataset:clone()
    local wmap = {}
    local y, idx = torch.topk(word_frequency, new_size, 1, true)
    local wf = torch.IntTensor(new_size)
    local unk_id
    local old_idx_to_new = {}
    for i=1,idx:size()[1] do
        old_idx_to_new[idx[i]] = i
        wf[i] = word_frequency[idx[i]]
        wmap[i] = word_map[idx[i]]
        if wmap[i] == '<UNK>' then
            unk_id = i
            print(unk_id)
        end
    end
    for i=1,ds:size()[1] do
        local word = old_idx_to_new[ds[i][2]]
        if wf[word] == nil then
            ds[i][2] = unk_id
            wf[unk_id] = wf[unk_id] + 1 -- one more word is now unknown
        else
            ds[i][2] = old_idx_to_new[ds[i][2]]
        end
    end
    for i=1,word_frequency:size()[1] do
        if old_idx_to_new[i] == nil then
            wf[unk_id] = wf[unk_id] + word_frequency[i]
        end
    end
    return ds, wmap, wf
end

-- reads a dataset and separates it into batches of size 50 or less, with sequence lengths all being the same in a batch
function bucket_training_set(dataset, max_unk_count, unk_value)
    local buckets = {}
    local batches = {}
    local occurences
    if max_unk_count ~= nil and unk_value ~= nil then
        occurences = count_occurences(dataset, unk_value)
    end
    local i = 1
    while i <= dataset:size()[1] do
        local sentence_id = dataset[i][1]
        local start = i
        while (i <= dataset:size()[1] and dataset[i][1] == sentence_id) do
            i = i + 1
        end
        local length = i - start - 1
        if occurences == nil or occurences[sentence_id] <= max_unk_count then
            local sentence = torch.IntTensor(length)
            local label = torch.IntTensor(length)
            for j=1,length do sentence[j] = dataset[start + j - 1][2] end
            for j=1,length do label[j] = dataset[start + j][2] end
            if buckets[length] == nil then
                buckets[length] = {{sentence,label}}
            else
                buckets[length][#(buckets[length]) + 1] = {sentence, label}
            end
        end
    end
    for seq_length, samples in pairs(buckets) do
        local i = 1
        while i <= #samples do
            local remaining = #samples - (i - 1)
            local batch = torch.IntTensor(math.min(50, remaining), seq_length)
            local labels = torch.IntTensor(math.min(50, remaining), seq_length)
            for j=1, batch:size()[1] do
                if i <= #samples then
                    batch[j] = samples[i][1]
                    labels[j] = samples[i][2]
                    i = i + 1
                end
            end
            batches[#batches + 1] = {batch, labels:reshape(math.min(50, remaining) * seq_length)}
        end
    end
    return batches
end

-- removes batches that aren't of exact batch size, and less than specified sequence length. Batch sizes must be consistent
-- when LSTM.remember_states is set to true.
function clean_dataset(t_set, batch_size, max_seq_length, tensorType)
    local trim_set = {}
    for k, v in pairs(t_set) do
        if type(v) ~= 'function' then
            if v[1]:size()[1] == batch_size and v[1]:size()[2] <= max_seq_length then
                trim_set[#trim_set + 1] = {v[1]:type(tensorType), v[2]:type(tensorType)}
            end
        end
    end
    function trim_set:size()
        return #trim_set
    end
    return trim_set
end

function count_occurences(raw_data_set, value)
    local result = torch.IntTensor(raw_data_set:size(1))
    local i = 1
    while i <= raw_data_set:size(1) do
        local count = 0
        local sent_id = raw_data_set[i][1]
        while i <= raw_data_set:size(1) and raw_data_set[i][1] == sent_id do
            if raw_data_set[i][2] == value then count = count + 1 end
            i = i + 1
        end
        result[sent_id] = count
    end
    return result
end

-- borrowed from: https://github.com/spro/torch-seq2seq-attention/blob/master/model_utils.lua
-- works like nn.Module():getParameters() but for multiple modules
function combine_all_parameters(...)
    --[[ like module:getParameters, but operates on many modules ]]--

    -- get parameters
    local networks = {...}
    local parameters = {}
    local gradParameters = {}
    for i = 1, #networks do
        local net_params, net_grads = networks[i]:parameters()

        if net_params then
            for _, p in pairs(net_params) do
                parameters[#parameters + 1] = p
            end
            for _, g in pairs(net_grads) do
                gradParameters[#gradParameters + 1] = g
            end
        end
    end

    local function storageInSet(set, storage)
        local storageAndOffset = set[torch.pointer(storage)]
        if storageAndOffset == nil then
            return nil
        end
        local _, offset = table.unpack(storageAndOffset)
        return offset
    end

    -- this function flattens arbitrary lists of parameters,
    -- even complex shared ones
    local function flatten(parameters)
        if not parameters or #parameters == 0 then
            return torch.Tensor()
        end
        local Tensor = parameters[1].new

        local storages = {}
        local nParameters = 0
        for k = 1,#parameters do
            local storage = parameters[k]:storage()
            if not storageInSet(storages, storage) then
                storages[torch.pointer(storage)] = {storage, nParameters}
                nParameters = nParameters + storage:size()
            end
        end

        local flatParameters = Tensor(nParameters):fill(1)
        local flatStorage = flatParameters:storage()

        for k = 1,#parameters do
            local storageOffset = storageInSet(storages, parameters[k]:storage())
            parameters[k]:set(flatStorage,
                storageOffset + parameters[k]:storageOffset(),
                parameters[k]:size(),
                parameters[k]:stride())
            parameters[k]:zero()
        end

        local maskParameters=  flatParameters:float():clone()
        local cumSumOfHoles = flatParameters:float():cumsum(1)
        local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
        local flatUsedParameters = Tensor(nUsedParameters)
        local flatUsedStorage = flatUsedParameters:storage()

        for k = 1,#parameters do
            local offset = cumSumOfHoles[parameters[k]:storageOffset()]
            parameters[k]:set(flatUsedStorage,
                parameters[k]:storageOffset() - offset,
                parameters[k]:size(),
                parameters[k]:stride())
        end

        for _, storageAndOffset in pairs(storages) do
            local k, v = table.unpack(storageAndOffset)
            flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
        end

        if cumSumOfHoles:sum() == 0 then
            flatUsedParameters:copy(flatParameters)
        else
            local counter = 0
            for k = 1,flatParameters:nElement() do
                if maskParameters[k] == 0 then
                    counter = counter + 1
                    flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
                end
            end
            -- assert (counter == nUsedParameters)
        end
        return flatUsedParameters
    end

    -- flatten parameters and gradients
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)

    -- return new flat vector that contains all discrete parameters
    return flatParameters, flatGradParameters
end

function no_ingredients(seq)
    local new_seq = torch.CudaTensor(seq:size(1)); local x;
    for i=1, seq:size(1) do
        new_seq[i] = helper.w_to_n['<pad>']
        if seq[i] == helper.w_to_n['<begin_ingredients>'] then x = i end
    end
    for i=1, x-1 do
        new_seq[new_seq:size(1) + 1 - i] = seq[x - i]
    end
    return new_seq
end