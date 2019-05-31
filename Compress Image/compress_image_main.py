from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


print('Running K-Means clustering on pixels from an image.\n\n')

image_name = 'sunset.png'


A = (plt.imread(image_name))
# imread returns an array A.shape = (row, column, RGB)
#rgb_rgba = 3 for rgb, 4 for rgba
row, column, rgb_rgba = A.shape


# display the image
plt.imshow(A)
plt.show()

# reshape A into an array A.shape = (n,3), n = # of pixels, 3 for r,g,b
A_new = A.reshape((row*column), rgb_rgba)

# run k-means on A_new
# choose the number of clusters i.d. the number of colours
num_cluster = 5
kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(A_new)
# centroids = ([r,g,b of cluster1],[r,g,b of cluster2], etc...)
centroids = kmeans.cluster_centers_
# idx = each pixels's cluster-belonging label (i.e. pixel 1 belongs to cluster 1, so idx[0] = 1)
idx = kmeans.labels_
print(min(idx),max(idx))

print('\nApplying K-Means to perform image compression.1\n\n')

#for every pixel substitute its r,g,b coordinates with that of its centroid
A_new = centroids[idx,:]

# recover the original image shape
A_rec = A_new.reshape(np.shape(A))

#plot the compressed image
plt.imshow(A_rec)
plt.show()