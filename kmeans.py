import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

headlines = ("Sepal length", "Sepal width", "Petal length", "Petal width", "Species")
data = pd.read_csv('data.csv',
                   names=headlines)
data.drop(columns="Species", inplace=True)

sl = "Sepal length"
sw = "Sepal width"
pl = "Petal length"
pw = "Petal width"

customcmap = ListedColormap(["blue", "red", "green"])


# Centroid initiation
def initiate_centroids(k, data_df):
    # Selecting 3 random points from the DataFrame
    centroids_df = data_df.sample(k)
    return centroids_df


# Calculation distance of point to centroid
def distance_to_centroid(point, centroid):
    return np.square(np.sum((point - centroid) ** 2))


# Assigning to each point the nearest centroid and the distance to this centroid
def points_assignation_to_centroids(data_df, centroids_df):
    assignation = []
    assign_distances = []

    for point in range(data_df.shape[0]):
        distances = np.array([])
        for centroid in range(centroids_df.shape[0]):
            # Calculation distance of each point to all centroids
            distance = distance_to_centroid(data_df.iloc[point], centroids_df.iloc[centroid])
            distances = np.append(distances, distance)

        nearest_centroid = np.where(distances == np.amin(distances))[0].tolist()[0]
        nearest_centroid_distance = np.amin(distances)

        assign_distances.append(nearest_centroid_distance)
        assignation.append(nearest_centroid)
    return assignation, assign_distances


# Setting new coordinates for each centroid
def set_new_centroids(data_df):
    centroids_df = data_df.groupby("centroid").mean().reset_index(drop=True)
    return centroids_df


def kmeans(data_df, k):
    data_df_copy = data_df.copy()
    centroids_df = initiate_centroids(k, data_df_copy)
    distance = []
    distances = []
    condition = True
    number_of_iteration = 0

    while condition:
        data_df_copy['centroid'], distance = points_assignation_to_centroids(data_df_copy, centroids_df)
        distances.append(sum(distance))
        centroids_df = set_new_centroids(data_df_copy)

        if number_of_iteration > 0:
            # Stop condition in which the assignment of points to centroids doesn't change
            # (the closest distances to centroids doesn't change)
            if distances[number_of_iteration - 1] == distances[number_of_iteration]:
                condition = False
        number_of_iteration += 1

    return data_df_copy['centroid'], distance, centroids_df, number_of_iteration


def plot(data_df, centroids, attribute1, attribute2):
    plt.figure(figsize=(8, 6))
    plt.scatter(data_df[attribute1], data_df[attribute2], marker='o',
                c=data_df['centroid'].astype('category'),
                cmap=customcmap, s=60, alpha=0.8)
    plt.scatter(centroids[attribute1], centroids[attribute2],
                marker='D', s=220,
                c=[0, 1, 2], cmap=customcmap)
    plt.xlabel(attribute1, fontsize=14)
    plt.ylabel(attribute2, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def main():
    data['centroid'], data['distance'], centroids, iterations = kmeans(data, 3)
    plot(data, centroids, sl, sw)
    plot(data, centroids, sl, pl)
    plot(data, centroids, sl, pw)
    plot(data, centroids, sw, pl)
    plot(data, centroids, sw, pw)
    plot(data, centroids, pl, pw)


if __name__ == '__main__':
    main()
