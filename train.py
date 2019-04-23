import glob
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions


class MLP(chainer.Chain):

    def __init__(self, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, 50)  # n_in -> n_units
            self.l2 = L.Linear(None, 50)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

def main():
    # Data preperation
    files = sorted(glob.glob('images/*.jpg'))
    nb_data = len(files)
    nb_data_per_class = len(files) // 17
    labels = np.arange(nb_data) // nb_data_per_class
    indices = np.arange(nb_data)
    np.random.permutation(indices)
    nb_train = 8 * nb_data // 10
    dataset = [(files[i], labels[i]) for i in indices]
    
    model = L.Classifier(MLP(17))

    use_gpu = -1
    if use_gpu >= 0:
        chainer.backends.cuda.get_device_from_id(use_gpu).use()
        model.to_gpu()
    
    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train = chainer.datasets.LabeledImageDataset(dataset[:nb_train])
    test = chainer.datasets.LabeledImageDataset(dataset[nb_train:])
    batchsize = 32
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=use_gpu)
    trainer = training.Trainer(updater, (30, 'epoch'), out='./result')

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=use_gpu))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()
