# 手肘法选取K值

from sklearn.cluster import KMeans
import multiprocessing


def train_cluster(train_vecs, model_name=None, start_k=2, end_k=20):
    print('training cluster')
    SSE = []
    SSE_d1 = []  # sse的一阶导数
    SSE_d2 = []  # sse的二阶导数
    models = []  # 保存每次的模型
    for i in range(start_k, end_k):
        kmeans_model = KMeans(n_clusters=kmeans_clusters, n_jobs=multiprocessing.cpu_count(), )
        kmeans_model.fit(train_vecs)
        SSE.append(kmeans_model.inertia_)  # 保存每一个k值的SSE值
        print('{} Means SSE loss = {}'.format(i, kmeans_model.inertia_))
        models.append(kmeans_model)
    # 求二阶导数，通过sse方法计算最佳k值
    SSE_length = len(SSE)
    for i in range(1, SSE_length):
        SSE_d1.append((SSE[i - 1] - SSE[i]) / 2)
    for i in range(1, len(SSE_d1) - 1):
        SSE_d2.append((SSE_d1[i - 1] - SSE_d1[i]) / 2)

    best_model = models[SSE_d2.index(max(SSE_d2)) + 1]
    return best_model


train_cluster(d, None, 1, 10)
