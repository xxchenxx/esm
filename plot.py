from regex import E
import pickle
temp = pickle.load(open("S_0.pkl", "rb"))
a = temp['train_labels']
b = temp['test_labels']
import matplotlib.pyplot as plt

plt.hist(list(a)+list(b))
plt.savefig("S.png")