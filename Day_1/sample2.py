from sklearn.cluster import KMeans

ages = [[18], [20], [22], [25], [50], [55], [60]]

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(ages)

print("Cluster Assignment:")
for age, cluster in zip(ages, kmeans.labels_):
    print(f"Age {age[0]} -> Cluster {cluster}")