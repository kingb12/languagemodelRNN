--
-- Created by IntelliJ IDEA.
-- User: bking
-- Date: 2/21/17
-- Time: 10:02 AM
-- To change this template use File | Settings | File Templates.
--

function sample(encoder, decoder, enc_state, sequence, max_samples)
    if max_samples == nil then
        max_samples = 1
    end
    if enc_state == nil then
        enc_state = encoder:forward(sequence)
        sequence = torch.CudaTensor({helper.w_to_n['<beg>']}):reshape(1, 1)
    end
    local cb = torch.CudaTensor.zeros(torch.CudaTensor.new(), 1, enc_state:size(1))
    local addition = torch.zeros(sequence:size(1)):cuda()
    local output = torch.cat(sequence, addition , 2)
    local y = decoder:forward(cb, enc_state, sequence)
    local sampled = sampler:forward(y)
    for i=1, output:size(1) do output[i][output:size(2)] = sampled[output:size(2) - 1] end
    if max_samples == 1 or wmap[output[1][output:size(2)]] == '</S>' then
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

function generate_samples(data_set, outputs, num_samples)
    local results = {}
    if opt.max_gen_example_length > 0 then
        data_set = truncate_dataset(data_set, opt.max_gen_example_length)
    end
    print('Generating Samples...')
    for i = 1, num_samples do
        print('Sample ', i)
        local t_set_idx = (torch.random() % #data_set) + 1
        if t_set_idx > #data_set then t_set_idx = 1 end
        local example = data_set[t_set_idx]
        local example_no = torch.random() % example:size(1) + 1
        if example_no > example:size(1) then example_no = 1 end
        local x = example[example_no]
        x = x:reshape(1, x:size(1))
        local result = {}
        result['generated'] = sequence_to_string(sample(enc, dec, nil, x, opt.max_sample_length))
        result['gold'] = sequence_to_string(outputs[t_set_idx][example_no])
        results[#results + 1] = result
    end
    return results
end

-- calculate perplexity
function perplexity_over_dataset(enc, dec, enc_inputs, dec_inputs, in_lengths, out_lengths, outputs)
    local data_perplexity = 0
    for i=1,enc_inputs:size(1) do
        for _,v in pairs(enc._rnns) do v:resetStates() end
        for _,v in pairs(dec._rnns) do v:resetStates() end
        local enc_input = enc_inputs[i]
        local dec_input = dec_inputs[i]
        local output = outputs[i]
        local enc_fwd = enc:forward(enc_input) -- enc_fwd is h1...hN
        local dec_h0 = enc_fwd[{{}, enc_inputs[1]:size(2), {}}] -- grab the last hidden state from the encoder, which will be at index max_in_len
        local dec_fwd = dec:forward({cb:clone(), dec_h0, dec_input}) -- forwarding a new zeroed cell state, the encoder hidden state, and frame-shifted expected output (like LM)
        dec_fwd = torch.reshape(dec_fwd, opt.batch_size, opt.max_out_len, opt.vocab_size)
        local loss = criterion:forward(dec_fwd, output) -- loss is essentially same as if we were a language model, ignoring padding
        loss = loss / (out_lengths[i] * dec_input:size(1)) -- normalize loss by # words in decoded ground truth and batch size
        local batch_perplexity = torch.exp(loss)
        data_perplexity = data_perplexity + (batch_perplexity / enc_inputs:size(1))
    end
    return data_perplexity
end

