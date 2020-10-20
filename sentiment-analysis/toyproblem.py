import numpy as np

from sklearn.neural_network import MLPClassifier
# epoch 705 optimized
# epoch 30 fo solution

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected = np.array([0,1,1,0])
for j in range(1, 9001):
    clf = MLPClassifier(hidden_layer_sizes=(6,),random_state=1, max_iter=j).fit(inputs, expected)

    accuracy = 0
    for i in range(len(inputs)):
        pred = clf.predict(inputs[i].reshape(1, -1))
        if pred[0] == expected[i]:
            accuracy += 1
    accuracy = accuracy / 4 * 100
    print(f'Accuracy of epoch {j} is {accuracy}%')
    if accuracy == 100:
        break

print(f"Succeeded after {j} epochs")
