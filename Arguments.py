
MODELFILE = 'model.pth'
OPTIMAIZER_ADAM = 'adam.opt'
OPTIMAZIER_SGD = 'sgd.opt'

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

class Arguments(BaseArgument):

    def __init__(self
        ,save_folder : str
        ,train_filepath : str
        ,valid_filepath : str
        ,is_use_adam = True
        ,gpu_id = -1
        ,batch_size = 256
        ,epochs = 10
        ,dropout = 0.3
        ,min_vocab_freq = 5
        ,max_vocab_size = 999999
        ,word_vec_size = 512
        ,hidden_size = 768
        ,max_length = 256
        ,layer_number = 4
        ,language = 'enko'
        ,off_autocast = False
        ,max_grad_norm = 5.0
        ,iteration_per_update = 1
        ,is_shutdown = False

    ):
        if save_folder[-1] != '/': save_folder += '/'
        super().__init__(
            save_folder+MODELFILE
            ,train_filepath
            ,gpu_id
            ,batch_size
            ,epochs
            ,dropout
            )
        self.save_folder = save_folder
        self.valid_filepath = valid_filepath
        self.optimfile = save_folder + OPTIMAIZER_ADAM if is_use_adam else save_folder + OPTIMAZIER_SGD
        self.min_vocab_freq = min_vocab_freq
        self.max_vocab_size = max_vocab_size
        self.word_vec_size = word_vec_size
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.layer_number = layer_number
        self.language = language
        self.off_autocast = off_autocast
        self.max_grad_norm = max_grad_norm
        self.iteration_per_update = iteration_per_update
        self.is_shutdown = is_shutdown


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


class NMTArgumets(BaseArgument):
    def __init__(self
        ,model_filepath : str
        ,train_filepath : str
        ,valid_filepath : str
        ,gpu_id = 0
        ,batch_size = 256
        ,init_epoch = 0
        ,epochs = 10
        ,dropout = 0.2
        ,lr = 1e-2
        ,lr_step = 1

        ,rl_lr = 0.01
        ,rl_n_samples = 1
        ,rl_init_epoch = -1
        ,rl_n_epochs = 0
        ,rl_n_gram = 6
        ,rl_reward = 'gleu'
        ,rl_batch_ratio = 1.

        ,max_grad_norm = 5.0
        ,word_vec_size = 256
        ,max_length = 256
        ,verbose = 2
        ,hidden_size = 256
        ,layer_number = 4
        ,language = 'enko'
        ,use_transformer=False
        ,n_splits = 8
        ,use_adam = True
        ,iteration_per_update = 1
        ,is_shutdonw = False
        ):
        super().__init__(
        model_filepath
        ,train_filepath
        ,gpu_id
        ,batch_size
        ,epochs
        ,dropout)

        self.valid_filepath = valid_filepath
        self.init_epoch = init_epoch
        self.lr = lr
        self.lr_step = lr_step
        
        self.rl_lr = rl_lr
        self.rl_n_samples = rl_n_samples
        self.rl_init_epoch = rl_init_epoch
        self.rl_n_epochs = rl_n_epochs
        self.rl_n_gram = rl_n_gram
        self.rl_reward = rl_reward
        self.rl_batch_ratio = rl_batch_ratio

        self.max_grad_norm = max_grad_norm
        self.word_vec_size = word_vec_size
        self.max_length = max_length
        self.verbose = verbose
        self.hidden_size = hidden_size
        self.layer_number = layer_number
        self.language = language
        self.use_transformer = use_transformer
        self.n_splits = n_splits
        self.use_adam = use_adam
        self.iteration_per_update = iteration_per_update
        self.is_shutdon = is_shutdonw


class NMTTranslateArg():
    def __init__(self
        ,model_filepath : str
        ,translate_filepath : str
        ,output_filepath : str
        ,batch_size = 128
        ,gpu_id = 0
        ,language = 'enko'
        ,beam_size = 1
        ,length_penalty = 1.2
    ):
        self.model_filepath = model_filepath
        self.translate_filepath = translate_filepath
        self.output_filepath = output_filepath
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self.language = language
        self.beam_size = beam_size
        self.length_penalty = length_penalty


