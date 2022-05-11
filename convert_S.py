import pickle
def t(x):
    x = float(x)
    if -20 <= x < 5:
        return 0
    elif 5 <= x < 25:
        return 1
    elif 25 <= x < 45:
        return 2
    elif 45 <= x < 75:
        return 3
    else:
        return 4

def t_binary(x):
    x = float(x)
    if x < 75:
        return 0
    else:
        return 1

for i in range(10):
# if True:
    pkl = pickle.load(open(f"d1_{i}.pkl", "rb"))
    train_labels = pkl['train_labels']
    test_labels = pkl['test_labels']
    train_labels = list(map(t_binary, train_labels))
    test_labels = list(map(t_binary, test_labels))

    pkl['train_labels'] = train_labels
    pkl['test_labels'] = test_labels

    pickle.dump(pkl, open(f"d1_{i}_classification.pkl", "wb"))
