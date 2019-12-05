import numpy as np
import matplotlib.pyplot as plt

knn_data_pca_train = np.array([
    [1, 1],
    [3, 0.919933],
    [5, 0.901917],
    [7, 0.893083],
    [9, 0.8874],
    [11, 0.882733],
    [13, 0.878417],
    [15, 0.875933],
    [17, 0.873517],
    [19, 0.870667],
])

knn_data_pca_test = np.array([
    [1, 0.842],
    [3, 0.852],
    [5, 0.8572],
    [7, 0.8595],
    [9, 0.8573],
    [11, 0.8564],
    [13, 0.8559],
    [15, 0.8564],
    [17, 0.8558],
    [19, 0.8537],
])

knn_data_lda_train = np.array([
    [1, 1],
    [3, 0.895317],
    [5, 0.88005],
    [7, 0.871933],
    [9, 0.868483],
    [11, 0.865317],
    [13, 0.8637],
    [15, 0.862217],
    [17, 0.861467],
    [19, 0.8601],
])

knn_data_lda_test = np.array([
    [1, 0.7911],
    [3, 0.814],
    [5, 0.8221],
    [7, 0.8279],
    [9, 0.8314],
    [11, 0.8325],
    [13, 0.8316],
    [15, 0.8322],
    [17, 0.831],
    [19, 0.8309],
])

data_train = np.array([
    [20, 0.798],
    [50, 0.814383],
    [80, 0.817267],
    [110, 0.81705],
    [140, 0.817283],
    [170, 0.814917],
    [200, 0.812733],
    [230, 0.811333],
    [260, 0.809133],
    [290, 0.807367],
    [320, 0.804833],
    [350, 0.803],
    [380, 0.800383],
    [410, 0.7989],
    [440, 0.796867],
    [470, 0.795933],
    [500, 0.795117],
    [530, 0.794583],
    [560, 0.79475],
    [590, 0.79505],
    [620, 0.795733],
    [650, 0.796483],
    [680, 0.79815],
    [710, 0.798133],
    [740, 0.798633],
    [770, 0.798767],
])

data_test = np.array([
    [20, 0.7887],
    [50, 0.8083],
    [80, 0.8053],
    [110, 0.8041],
    [140, 0.7984],
    [170, 0.7937],
    [200, 0.788],
    [230, 0.782],
    [260, 0.7764],
    [290, 0.7704],
    [320, 0.763],
    [350, 0.7562],
    [380, 0.7505],
    [410, 0.7401],
    [440, 0.7281],
    [470, 0.7162],
    [500, 0.7092],
    [530, 0.697],
    [560, 0.681],
    [590, 0.6711],
    [620, 0.6601],
    [650, 0.7288],
    [680, 0.7284],
    [710, 0.729],
    [740, 0.7285],
    [770, 0.7283],
])

plt.figure()
plt.plot(knn_data_pca_train[:, 0], knn_data_pca_train[:, 1], marker='o', color='r', label='train_after_PCA(dim:50)', linewidth=1)
plt.plot(knn_data_pca_test[:, 0], knn_data_pca_test[:, 1], marker='x', color='r', label='test_after_PCA(dim:50)', linewidth=1)
plt.plot(knn_data_lda_train[:, 0], knn_data_lda_train[:, 1], marker='o', color='b', label='train_after_LDA(dim:9)', linewidth=1)
plt.plot(knn_data_lda_test[:, 0], knn_data_lda_test[:, 1], marker='x', color='b', label='test_after_LDA(dim:9)', linewidth=1)

# plt.bar(data_train[:, 0], data_train[:, 1], label='train')
# plt.bar(data_test[:, 0], data_test[:, 1], label='test', color='r')

plt.xlabel("K Neighbors")
plt.ylabel("Accuracy")
plt.title("Accuracy of KNN")
plt.legend()
plt.grid()
plt.xticks(np.arange(1, 20, 2))
plt.show()
