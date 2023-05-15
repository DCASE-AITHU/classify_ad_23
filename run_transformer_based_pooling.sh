export PYTHONPATH=`pwd`:$PYTHONPATH

## first step: embedding extraction, kaldiio package needed
# results23exp_wav2vec_300m_2seval_interval200_0_condition_step_3600_method_knn_mean_hmean_63.842
# results23exp_hubert_large_2seval_interval200_1_condition_step_8600_method_knn_mean_hmean_61.743
# results23exp_unispeech_large_2seval_interval200_1_condition_step_9400_method_knn_mean_hmean_62.382
# results23exp_wavlm_large_2seval_interval200_2_condition_step_9800_method_knn_mean_hmean_61.401
ckpt=best_ckpts/results23exp_wavlm_large_2seval_interval200_2_condition_step_9800_method_knn_mean_hmean_61.401
prefix=wavlm_large_2seval_interval200_2_condition_step_9800_method_knn_mean_hmean_61.401
device=4
model="wavlm_large" # model type, defined in w2v.py 
output_path=arks
input_duration=2
shift_duration=0.5
dataset=2 #0:dev 1:eval 2:dev+eval
# for mobilefacenet model, using extract_embedding.py, but with less parameters
CUDA_VISIBLE_DEVICES=$device python extract_embedding_w2v.py --dataset $dataset \
                                                             -or $output_path \
                                                             -duration $input_duration \
                                                             -shift $shift_duration \
                                                             -ckpt $ckpt \
                                                             -prefix $prefix \
                                                             -model $model

# second step: tranformer-based-pooling
scp=$output_path/$prefix.scp
n_times=3 # 3 individual runs and append output embedding
result_path=csv_path
domsp=none # none, hard, soft, min
CUDA_VISIBLE_DEVICES=$device python aggregation/main_multi.py --ark_scp $scp \
                                                              --n_times ${n_times} \
                                                              --result_path $result_path \
                                                              --domsp $domsp
