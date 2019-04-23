import glob
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions


class SimpleNet(chainer.Chain):

    def __init__(self, n_out):
        super(SimpleNet, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(None, 32, 7)
            self.b1 = L.BatchNormalization(32)
            self.c2 = L.Convolution2D(None, 64, 5)
            self.b2 = L.BatchNormalization(64)
            self.c3 = L.Convolution2D(None, 64, 3)
            self.b3 = L.BatchNormalization(64)
            self.fc4 = L.Linear(None, 256)
            self.fc5 = L.Linear(None, n_out)
            
    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = F.relu(self.b2(self.c2(h)))
        h = F.relu(self.b3(self.c3(h)))
        h = F.relu(self.fc4(h))
        return self.fc5(h)


#def main():
def handler(context):
    # Data preperation
    files = sorted(glob.glob('/mnt/notebooks/1745249968478/images1/*.jpg'))
    nb_data = len(files)
    nb_data_per_class = len(files) // 17
    labels = np.arange(nb_data) // nb_data_per_class
    indices = np.arange(nb_data)
    indices = np.random.permutation(indices)
    nb_train = 8 * nb_data // 10
    dataset = [(files[i], labels[i]) for i in indices]
    
    model = L.Classifier(SimpleNet(17))

    use_gpu = 0
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
