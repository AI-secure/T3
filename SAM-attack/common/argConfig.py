import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch TreeLSTM for Sentence Similarity on Dependency Trees')
    parser.add_argument('--mode', default='train',
                        help='network mode')
    parser.add_argument('--logger', default='generator',
                        help='logger name')
    # data arguments
    parser.add_argument('--data', default='./data/imdb/',
                        help='path to dataset')
    parser.add_argument('--glove', default='./data/glove/',
                        help='directory with GLOVE embeddings')
    parser.add_argument('--save', default='./checkpoints/',
                        help='directory to save checkpoints in')
    parser.add_argument('--save_seqback', type=str, default='./checkpoints/seqback_model.pt',
                        help='path to save the seqback LSTM model')
    parser.add_argument('--save_end2end', type=str, default='./checkpoints/',
                        help='path to save the end to end model')
    parser.add_argument('--save_ans', type=str, default=None,
                        help='path to save the answer')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='path to load the data')
    parser.add_argument('--global_dir', type=str, default="data",
                        help='path to load the global data')
    parser.add_argument('--model', type=str, default="seq2seq",
                        help='generator model')
    parser.add_argument('--expname', type=str, default='test',
                        help='Name to identify experiment')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Name to load pre-trained model')
    parser.add_argument('--load_data', type=bool, default=True,
                        help='Load presaved data')
    # model arguments
    parser.add_argument('--input_dim', default=300, type=int,
                        help='Size of input word vector')
    parser.add_argument('--mem_dim', default=300, type=int,
                        help='Size of TreeLSTM cell state')
    parser.add_argument('--hidden_dim', default=300, type=int,
                        help='Size of classifier MLP')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='Number of classes in dataset')
    parser.add_argument('--freeze_embed', action='store_true',
                        help='Freeze word embeddings')
    parser.add_argument('--encode_rel', default=True, type=bool,
                        help='TreeLSTM encoding has relation.')
    parser.add_argument('--decode_word', default=True, type=bool,
                        help='TreeLSTM decodes with word.')
    # training arguments
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--test_train', action='store_true',
                        help='do evaluation on training set')
    parser.add_argument('--eval_func', action='store_true',
                        help='eval function at each step')
    parser.add_argument('--debugging', action='store_true',
                        help='output debug at each step')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batchsize for optimizer updates')
    parser.add_argument('--lr', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--tr', default=1.0, type=float,
                        help='default teacher ratio')
    parser.add_argument('--temp', default=1e-1, type=float,
                        help='softmax temparature')
    parser.add_argument('--sparse', action='store_true',
                        help='Enable sparsity for embeddings, \
                              incompatible with weight decay')
    parser.add_argument('--optim', default='adagrad',
                        help='optimizer (default: adagrad)')
    # miscellaneous options
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed (default: 123)')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args
