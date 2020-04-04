from data import dataLoader
from source import metrics
import _pickle as pkl

test_x, test_y = dataLoader.load_data('test')
with open('model/classifier.pkl', 'rb') as file:
    classifier = pkl.load(file)

n_samples = test_x.shape[0]
acc_counter = metrics.Accuracy()
f1_counter = metrics.MacroF1(dataLoader.NUM_LABEL)

for x, y in zip(test_x, test_y):
    pred_y = classifier.predict(x)
    acc_counter.add_sample(pred_y, y)
    f1_counter.add_sample(pred_y, y)

acc = acc_counter.calculate()
f1 = f1_counter.calculate()
print("The test acc is {} and the test Macro-F1 score is {}".format(acc, f1))
