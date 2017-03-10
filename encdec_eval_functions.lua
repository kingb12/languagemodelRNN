--
-- Created by IntelliJ IDEA.
-- User: bking
-- Date: 2/21/17
-- Time: 10:02 AM
-- To change this template use File | Settings | File Templates.
--

cjson = require 'cjson'

function sample(encoder, decoder, enc_state, sequence, max_samples)
    if max_samples == nil then
        max_samples = 1
    end
    if enc_state == nil then
        enc_state = encoder:forward(sequence)
        sequence = torch.CudaTensor({helper.w_to_n['<beg>']}):reshape(1, 1)
    end
    local cb = torch.CudaTensor.zeros(torch.CudaTensor.new(), 1, enc_state:size(3))
    local addition = torch.zeros(sequence:size(1)):cuda()
    local output = torch.cat(sequence, addition , 2)
    local dec_h0 = enc_state[{{}, enc_state:size(2), {}}] -- grab the last hidden state from the encoder, which will be at index max_in_len
    local y = decoder:forward({cb:clone(), dec_h0, sequence})
    local sampled = sampler:forward(y)
    for i=1, output:size(1) do output[i][output:size(2)] = sampled[output:size(2) - 1] end
    if max_samples == 1 or helper.n_to_w[output[1][output:size(2)]] == '</S>' then
        return output
    else
        return sample(encoder, decoder, enc_state, output, max_samples - 1)
    end
end

function sequence_to_string(seq)
    local str = ''
    if seq:dim() == 2 then seq = seq[1] end
    for i=1, seq:size()[1] do
        local next_word = helper.n_to_w[seq[i]] == nil and '<UNK2>' or helper.n_to_w[seq[i]]
        str = str..' '..next_word
    end
    return str
end

function generate_samples(data_set, outputs, num_samples, max_sample_length)
    if max_sample_length == nil then max_sample_length = 10 end
    local results = {}
    print('Generating Samples...')
    for i = 1, num_samples do
        print('Sample ', i)
        local t_set_idx = (torch.random() % data_set:size(1)) + 1
        if t_set_idx > data_set:size(1) then t_set_idx = 1 end
        local example = data_set[t_set_idx]
        local example_no = torch.random() % example:size(1) + 1
        if example_no > example:size(1) then example_no = 1 end
        local x = example[example_no]
        x = x:reshape(1, x:size(1))
        local result = {}
        result['encoder_input'] = sequence_to_string(x)
        result['generated'] = sequence_to_string(sample(enc, dec, nil, x, max_sample_length))
        result['gold'] = sequence_to_string(outputs[t_set_idx][example_no])
        results[#results + 1] = result
    end
    return results
end

-- calculate perplexity
function perplexity_over_dataset(enc, dec, enc_inputs, dec_inputs, in_lengths, out_lengths, outputs)
    local cb = torch.CudaTensor.zeros(torch.CudaTensor.new(), enc_inputs[1]:size(1), enc:forward(enc_inputs[1]):size(3))
    local data_perplexity = 0
    local data_loss = 0
    for i=1,enc_inputs:size(1) do
        for _,v in pairs(enc._rnns) do v:resetStates() end
        for _,v in pairs(dec._rnns) do v:resetStates() end
        local enc_input = enc_inputs[i]
        local dec_input = dec_inputs[i]
        local output = outputs[i]
        local enc_fwd = enc:forward(enc_input) -- enc_fwd is h1...hN
        local dec_h0 = enc_fwd[{{}, enc_inputs[1]:size(2), {}}] -- grab the last hidden state from the encoder, which will be at index max_in_len
        local dec_fwd = dec:forward({cb:clone(), dec_h0, dec_input}) -- forwarding a new zeroed cell state, the encoder hidden state, and frame-shifted expected output (like LM)
        dec_fwd = torch.reshape(dec_fwd, enc_input:size(1), dec_input:size(2), #helper.n_to_w)
        local loss = criterion:forward(dec_fwd, output) -- loss is essentially same as if we were a language model, ignoring padding
        loss = loss / (torch.sum(out_lengths[i]) / enc_inputs[i]:size(1))
        local batch_perplexity = torch.exp(loss)
        data_perplexity = data_perplexity + (batch_perplexity / enc_inputs:size(1))
        data_loss = data_loss + (loss / enc_inputs:size(1))
    end
    return data_perplexity, data_loss
end

function cmdout (cmd)
   local f = io.popen(cmd)
   local lout = f:read("*a")
   f:close()
   return lout
end

function calculate_bleu(references, candidates)
   if reference == nil then reference = 'reference.txt' end
   if candidate == nil then candidate = 'candidate.txt' end
    local io = require 'io'
    local f = io.open(reference, 'w+')
    for i=1, #references do
        f:write(references[i] .. '\n')
    end
    f:close()
    local f = io.open(candidate, 'w+')
    for i=1, #candidates do
        f:write(candidates[i] .. '\n')
    end
    f:close()
    local cmd = 'perl multi-bleu.perl ' .. reference ..' < ' .. candidate .. ' | python parse_bleu.py'
    local s = cmdout(cmd)
    return cjson.decode(s)
end

function n_pairs_bleu(generations, n)
    local refs = {}
    local cands = {}
    for i=1, n do
        local i = (torch.random() % #generations) + 1
        local j = (torch.random() % #generations) + 1
        while i == j and #generations > 1 do
            j = (torch.random() % #generations) + 1
        end
        refs[#refs + 1] = generations[i]:gsub('\'', '\\\'')
        cands[#cands + 1] = generations[j]:gsub('\'', '\\\'')

    end
    return calculate_bleu(refs, cands)
end

function closest_bleu_match(references, sample)
    local best_score = 0
    local best_match = ''
    for i=1, #references do
        -- literally writing a file for each reference, so definitely room for performance improvement here
        local scores = calculate_bleu({references[i]}, {sample})
        if scores['score'] > best_score then
            best_score = scores['score']
            best_match = references[i]
        end
        if scores['score'] == 100 then
            break
        end
    end
    return best_match, best_score
end

function alignment_scores(sequences)
    local cmd = 'python alignment.py ' .. '\'' .. cjson.encode(sequences):gsub('\'', '') .. '\''
    local s = cmdout(cmd)
    local t = cjson.decode(s)
    local data = torch.DoubleTensor(t)
    local avg_alignment_score = 0; local count = 0
    for i=1, #sequences - 1 do for j=i + 1, #sequences do
        avg_alignment_score = avg_alignment_score + data[i][j]; count = count + 1
    end end
    avg_alignment_score = avg_alignment_score / count
    return avg_alignment_score
end
