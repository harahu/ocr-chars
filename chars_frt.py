from sklearn.ensemble import RandomForestClassifier
import chars_dataset
import numpy as np

# Import data
chars = chars_dataset.read_data_sets()

X = chars.train.images
Y = chars.train.labels
X_ = chars.test.images
Y_ = chars.test.labels
clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
clf = clf.fit(X, Y)

Yc = clf.predict(X_)
correct_prediction = np.equal(np.argmax(Yc, 1), np.argmax(Y_, 1))
accuracy = np.mean(correct_prediction.astype(np.float32))

print(accuracy)