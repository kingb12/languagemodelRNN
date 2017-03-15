# Template for running experiments.

# meta options
save_dir='/scratch/kingb12/' 
model_name='encdec_noing10_bow_200_512_04drbef'
# common adjustments
max_epochs=200
learning_rate=0.00001
num_samples=7
max_sample_length=25
##################################################### Model Run Options #############################################
# Dataset options
enc_inputs='../data/rl_no_ing_10.th7'
dec_inputs='../data/rl_dec_inputs_25.th7'
outputs='../data/rl_outputs_25.th7'
in_lengths='../data/rl_in_lengths.th7'
out_lengths='../data/rl_out_lengths_25.th7'
helper='../data/rl_helper.th7'
gen_inputs='[4,209,512,1896,1743,143,865,443,30,201]'


valid_enc_inputs='/homes/iws/kingb12/data/rl_vno_ing_10.th7'
valid_dec_inputs='/homes/iws/kingb12/data/rl_vdec_inputs_25.th7'
valid_outputs='/homes/iws/kingb12/data/rl_voutputs_25.th7'
valid_in_lengths='/homes/iws/kingb12/data/rl_vin_lengths.th7'
valid_out_lengths='/homes/iws/kingb12/data/rl_vout_lengths_25.th7'

test_enc_inputs='/homes/iws/kingb12/data/rl_tno_ing_10.th7'
test_dec_inputs='/homes/iws/kingb12/data/rl_tdec_inputs_25.th7'
test_outputs='/homes/iws/kingb12/data/rl_toutputs_25.th7'
test_in_lengths='/homes/iws/kingb12/data/rl_tin_lengths.th7'
test_out_lengths='/homes/iws/kingb12/data/rl_tout_lengths_25.th7'

max_in_len=10
max_out_len=25
min_out_len=1
min_in_len=1
batch_size=4

init_enc_from=''
init_dec_from=''
wordvec_size=200
hidden_size=512
dropout=0.4
dropout_loc='both'
num_enc_layers=1
num_dec_layers=1
weights=''

lr_decay=0.0
algorithm='adam'

print_loss_every=1000
save_prefix=$save_dir$model_name'/'$model_name
backup_save_dir=''
print_acc_every=0
print_examples_every=0
valid_loss_every=1

######################################################### Evaluation Options ##############################################

max_gen_example_length=10
out=$save_prefix'.json'

######################################################## Actually Running #################################################

cd ..

th encdec_evaluation.lua \
-train_enc_inputs $enc_inputs \
-train_dec_inputs $dec_inputs \
-train_outputs $outputs \
-train_in_lengths $in_lengths \
-train_out_lengths $out_lengths \
-helper $helper \
-init_output_from $out \
-enc $save_prefix'_enc.th7' \
-dec $save_prefix'_dec.th7' \
-valid_enc_inputs $valid_enc_inputs \
-valid_dec_inputs $valid_dec_inputs \
-valid_outputs $valid_outputs \
-valid_in_lengths $valid_in_lengths \
-valid_out_lengths $valid_out_lengths \
-test_enc_inputs $test_enc_inputs \
-test_dec_inputs $test_dec_inputs \
-test_outputs $test_outputs \
-test_in_lengths $test_in_lengths \
-test_out_lengths $test_out_lengths \
-num_samples $num_samples \
-max_sample_length $max_sample_length \
-gen_inputs $gen_inputs -generate_samples \
-calculate_bleu -calculate_avg_alignment -calculate_n_pairs_bleu \
-calculate_best_bleu_match  \
-out $out
