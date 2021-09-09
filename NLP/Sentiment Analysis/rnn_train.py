import sys
import os

from train import train_run
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Arguments import NLPArgument as narg

if __name__ == '__main__':
    arg = narg(model_filepath='Data/Models/rnn_review.pth', 
                    train_filepath='Data/review.sorted.uniq.refined.tsv',
                    epochs=10,
                    gpu_id=0 ,
                    batch_size=512,

                    use_rnn=True, 
                    hidden_size = 512,
                    layer_number = 4,
                    
                    use_cnn=False)
    train_run(arg)