# classify_ad_23
classification baseline (either machine ids or different work conditions) by yufengdeng

# How to run
modify task in runNtimes.py, choosing to classify machines or conditions

modify nTimes to decide how many runs for averaging

modify max_step to control how many steps to train


Then run "python runNtimes.py"

# Results
Rs-{expname}-mt{mt}-{curT}-{task}.csv in format "machine, s_auc_mean, s_auc_std, t_auc_mean, t_auc_std, pauc_mean, pauc_std" in 5 runs

# Bug
There are a lot of same audios tagged with "anomaly" and "normal" in development test dataset, which leads to wrong conclusions now.
|Machine| Repeat|
| ----------- | ----------- |
|ToyCar| 0|
|ToyTrain|0|
|bearing|26|
|fan|50|
|gearbox|28|
|slider|30|
|valve|49|
