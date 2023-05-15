import os
import glob
import pandas as pd

EVAL_TYPES23 = [
    'bandsaw',
    'grinder',
    'shaker',
    'ToyDrone',
    'ToyNscale',
    'ToyTank',
    'Vacuum'
]

df = pd.read_csv('dom_pred/mfn_pred_mtdom_dev.csv', index_col=False)
df.drop('Unnamed: 0', axis=1, inplace=True)
for mt in EVAL_TYPES23:
    files = glob.glob(f'/home/public/dcase/dcase23/updated/eval_data/{mt}/test/*.wav')
    mt_df = pd.DataFrame({'name': [], 'mt': [], 'domain': [],
                          'source': [], 'target': []})
    for file in files:
        file_id = os.path.basename(file)
        domain = -1
        mt_df.loc[len(mt_df)] = [file_id, mt, domain, 0.5, 0.5]
    df = pd.concat([df, mt_df], axis=0, ignore_index=True)
df.to_csv('dom_pred/mfn_pred_mtdom.csv', index=False)
