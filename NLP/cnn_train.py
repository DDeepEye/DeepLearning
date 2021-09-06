import sys
import os

from train import train_run
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Arguments import NLPArgument

if __name__ == '__main__':
    arg = NLPArgument(model_filepath='Data/Models/cnn_review.pth', 
                    train_filepath='Data/review.sorted.uniq.refined.tsv',
                    gpu_id=-1 ,
                    batch_size=512 ,

                    use_cnn=True, 
                    window_sizes=[3,4,5,6,7,8], 
                    filter_sizes=[128,128,128,128,128,128],
                    use_rnn=False)
    train_run(arg)