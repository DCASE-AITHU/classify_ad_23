export PYTHONPATH=`pwd`:$PYTHONPATH

## first step: embedding extraction, kaldiio package needed
ckpt=best_ckpts/results23exp_wav2vec_300m_2seval_interval200_0_condition_step_3600_method_knn_mean_hmean_63.842
prefix=wav2vec_300m_2seval_interval200_0_condition_step_3600_method_knn_mean_hmean_63.842
device=0
model="wav2vec_300m" # model type, defined in w2v.py 
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

## second step: tranformer-based-pooling
scp=$output_path/$prefix.scp
n_times=3 # 3 individual runs and append output embedding
result_path=csv_path
CUDA_VISIBLE_DEVICES=$device python aggregation/main_multi.py  $scp  ${n_times} $result_path
