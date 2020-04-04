from data import dataLoader
from tqdm import tqdm
from source.model import LogLinear
from source import metrics
import numpy as np
import _pickle as pkl

training_x, training_y = dataLoader.load_data()
valid_x, valid_y = dataLoader.load_data('valid')
NUM_EPOCH = 50
BATCH_SIZE = 32
n_train = training_x.shape[0]

x_indices = np.arange(0, n_train)
x_indices.dtype = 'int'

classifier = LogLinear(dataLoader.NUM_LABEL, dataLoader.NUM_SAMPLE, 0.1)

acc_counter = metrics.Accuracy()
f1_counter = metrics.MacroF1(dataLoader.NUM_LABEL)
train_endurance = 0
update_endurance = 0
acc0 = 0

for epoch in range(NUM_EPOCH):
    np.random.shuffle(x_indices)
    training_x = training_x[x_indices]
    training_y = training_y[x_indices]

    for i in tqdm(range(0, n_train, BATCH_SIZE)):
        end = i + BATCH_SIZE if i + BATCH_SIZE < n_train else n_train
        x = training_x[i: end]
        y = training_y[i: end]
        avg_loss = classifier.train(x, y, True)

    train_acc = metrics.Accuracy()
    train_f1 = metrics.MacroF1(dataLoader.NUM_LABEL)
    for x, y in zip(training_x, training_y):
        prediction = classifier.predict(x)
        train_acc.add_sample(prediction, y)
        train_f1.add_sample(prediction, y)
    acc = train_acc.calculate()
    f1 = train_f1.calculate()
    print("In epoch {}, the training acc is {}, the training Macro-F1 score is {}".format(epoch, acc, f1))

    acc_counter.reset()
    f1_counter.reset()
    for x, y in zip(valid_x, valid_y):
        prediction = classifier.predict(x)
        acc_counter.add_sample(prediction, y)
        f1_counter.add_sample(prediction, y)

    acc = acc_counter.calculate()
    f1 = f1_counter.calculate()
    print("In epoch {}, the acc is {}, the Macro-F1 score is {}".format(epoch, acc, f1))
    if acc < acc0 or np.abs(acc - acc0) < 1e-3:
        train_endurance += 1
        update_endurance += 1
        if update_endurance > 2:
            classifier.set_lr(classifier.lr / 2)
            update_endurance = 0
        if train_endurance > 5:
            break
    else:
        acc0 = acc
        train_endurance = 0
        update_endurance = 0
        with open('model/classifier.pkl', 'wb') as file:
            pkl.dump(classifier, file)
