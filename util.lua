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

-- TODO: Write a function to parse tables to Tensors, specify label index

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

--
function reduce_vocab_size(dataset, word_map, word_frequency, new_size)
    -- read the whole vocab table and reverse it to (freq -> {indexes})
    -- take the n most frequent indexes, build a new wmap, wfreq where freq(<UNK>) sum(word_freq) - sum(new)
    -- save it as a new dataset, new word_freq, new wmap
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

function bucket_training_set(dataset)
    local buckets = {}
    local batches = {}
    local i = 1
    while i <= dataset:size()[1] do
        local sentence_id = dataset[i][1]
        local start = i
        while (i <= dataset:size()[1] and dataset[i][1] == sentence_id) do
            i = i + 1
        end
        local length = i - start - 1
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
    for seq_length, samples in pairs(buckets) do
        local i = 1
        while i <= #samples do
            local remaining = #samples - (i - 1)
            local batch = torch.IntTensor(math.min(50, remaining), seq_length)
            local labels = torch.IntTensor(math.min(50, remaining), seq_length)
            for j=1, batch:size()[1] - 1 do
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
