
class BaseArgument():
    def __init__(self,
        model_filepath : str,
        train_filepath : str,
        gpu_id = -1,
        batch_size = 256,
        epochs = 10,
        dropout = 0.3,
        ):
        self.model_filepath = model_filepath
        self.train_filepath = train_filepath
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        


class NLPArgument(BaseArgument):
    def __init__(self
        ,model_filepath : str
        ,train_filepath : str
        ,gpu_id = -1
        ,batch_size = 256
        ,epochs = 10
        ,dropout = 0.3
        ,min_vocab_freq = 5
        ,max_vocab_size = 999999
        ,word_vec_size = 256
        ,max_length = 256

        ,use_rnn = True
        ,hidden_size = 256
        ,layer_number = 4

        ,use_cnn = True
        ,use_batch_norm = True
        ,window_sizes = [3,4,5]
        ,filter_sizes = [128,128,128]
        ):

        super().__init__(
        model_filepath
        ,train_filepath
        ,gpu_id
        ,batch_size
        ,epochs
        ,dropout)

        self.min_vocab_freq = min_vocab_freq
        self.max_vocab_size = max_vocab_size
        self.word_vec_size = word_vec_size
        self.max_length = max_length

        self.use_rnn = use_rnn
        self.hidden_size = hidden_size
        self.layer_number = layer_number

        self.use_cnn = use_cnn
        self.use_batch_norm = use_batch_norm
        self.window_sizes = window_sizes
        self.filter_sizes = filter_sizes
        