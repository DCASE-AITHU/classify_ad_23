from kaldiio import ReadHelper
from kaldiio import WriteHelper
import numpy as np
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:{} scp_list output_prefix".format(sys.argv[0]))
        exit(0)
    
    mats = {}
    with open(sys.argv[1]) as fin:
        for scp in fin.readlines():
            with ReadHelper('scp:'+scp.strip()) as reader:
                for key, mat in reader:
                    if key not in mats:
                        mats[key] = []
                    mats[key].append(mat)
    with WriteHelper('ark,scp:{prefix}.ark,{prefix}.scp'.format(prefix=sys.argv[2])) as writer:
        for key in mats.keys():
            writer(key, np.hstack(mats[key]))

