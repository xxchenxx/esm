import pickle
def t(x):
    x = float(x)
    if -20 <= x < 5:
        return 0
    elif 5 <= x < 25:
        return 0
    elif 25 <= x < 45:
        return 1
    elif 45 <= x < 75:
        return 1
    else:
        return 1
# for i in range(10):
if True:
    pkl = pickle.load(open(f"S_target.pkl", "rb"))
    train_labels = pkl['train_labels']
    test_labels = pkl['test_labels']
    train_labels = list(map(t, train_labels))
    test_labels = list(map(t, test_labels))

    pkl['train_labels'] = train_labels
    pkl['test_labels'] = test_labels

    pickle.dump(pkl, open(f"S_target_new_classification.pkl", "wb"))
