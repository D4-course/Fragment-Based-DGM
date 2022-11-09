"""Main """
from rdkit import rdBase
from learner.dataset import FragmentDataset
from learner.sampler import Sampler
from learner.trainer import Trainer, save_ckpt
from utils.config import Config
from utils.parser import command_parser
from utils.plots import plot_paper_figures
from utils.preprocess import preprocess_dataset
from utils.postprocess import postprocess_samples, score_samples, dump_scores
from utils.filesystem import load_dataset

rdBase.DisableLog('rdApp.*')



def train_model(config):
    """train_model"""
    print(config)
    dataset = FragmentDataset(config)
    vocab = dataset.get_vocab()
    trainer = Trainer(config, vocab)
    trainer.train(dataset.get_loader(), 0)


def resume_model(config):
    """resume_model"""
    dataset = FragmentDataset(config)
    vocab = dataset.get_vocab()
    load_last = config.get('load_last')
    trainer, epoch = Trainer.load(config, vocab, last=load_last)
    trainer.train(dataset.get_loader(), epoch + 1)


def sample_model(config):
    """sample_model"""
    dataset = FragmentDataset(config)
    vocab = dataset.get_vocab()
    load_last = config.get('load_last')
    print(load_last)
    trainer, epoch = Trainer.load(config, vocab, last=load_last)
    sampler = Sampler(config, vocab, trainer.model)
    seed = config.get('sampling_seed') if config.get('reproduce') else None
    samples = sampler.sample(config.get('num_samples'), seed=seed)
    dataset = load_dataset(config, kind="test")
    _, scores = score_samples(samples, dataset)
    is_max = dump_scores(config, scores, epoch)
    if is_max:
        save_ckpt(trainer, epoch, filename=f"best.pt")
    config.save()


if __name__ == "__main__":
    parser = command_parser()
    args = vars(parser.parse_args())
    command = args.pop('command')

    if command == 'preprocess':   # this step downloads the requested datset and processes it by adding additional information for each molecule like fragments, no of bonds, no of rings and no of bonds 
        dataset = args.pop('dataset')
        n_jobs = args.pop('n_jobs')
        preprocess_dataset(dataset, n_jobs)

    elif command == 'train':   # this step takes the postprocessed datset as input and does the training part, where embeddings are done and each molecule is sent through the VAE, num_epochs = 20
        config = Config(args.pop('dataset'), **args)
        train_model(config)

    elif command == 'resume':   # this step is used for resuming training in the case of pre-eptively stopping the training process
        run_dir = args.pop('run_dir')
        config = Config.load(run_dir, **args)
        resume_model(config)

    elif command == 'sample':   # Generates sample molecules from the decoder part of the trained model
        args.update(use_gpu=False)
        run_dir = args.pop('run_dir')
        config = Config.load(run_dir, **args)
        sample_model(config)

    elif command == 'postprocess':   # Generates statistics from sampled data and aggregates multiple sample files and the test data in one big file for plotting
        run_dir = args.pop('run_dir')
        config = Config.load(run_dir, **args)
        postprocess_samples(config, **args)

    elif command == 'plot':   # Generates figures shown in the paper with the help of data from the previous steps
        run_dir = args.pop('run_dir')
        plot_paper_figures(run_dir)
