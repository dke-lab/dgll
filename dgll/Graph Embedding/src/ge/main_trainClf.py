import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy.sparse as sp
import ge
# Load Embedding
print("Load Embedding")
idsEmbeddingClsLabels = np.genfromtxt("../data/cora.embedding", dtype=np.dtype(str))

print("Classifier Training")
# Prepare data for training testing
labels = idsEmbeddingClsLabels[:, -1]
embedding = sp.csr_matrix(idsEmbeddingClsLabels[:, 1:-1], dtype=np.float32)


tr = ge.TrainingClassifiers()
y = tr.labelEnocder(labels)
# Prepare Train test data
X_train, X_test, y_train, y_test = tr.prepareTrainTestData(embedding, labels, 0.33)

# Choose one of the following classifier for training train a classifier

# y_pred = tr.applyDecisionTree(X_train.toarray(), y_train, X_test.toarray())
# y_pred = tr.applyLogistic(X_train.toarray(), y_train, X_test.toarray())
# y_pred = tr.applyRandomForest(X_train.toarray(), y_train, X_test.toarray())
# y_pred = tr.apply_GradientBoosting(X_train.toarray(), y_train, X_test.toarray())

y_pred = tr.applyMLP(X_train.toarray(), y_train, X_test.toarray())
print("Accuracy:", tr.accuracy(y_test, y_pred))
