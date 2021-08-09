# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

column_list = ["category_id", "views", "likes", "dislikes", "comment_count", "comments_disabled", "ratings_disabled", "video_error_or_removed"]
data = pd.read_csv("USvideos.csv", usecols=column_list)
data = data.groupby("category_id").apply(lambda x: x.sample(50)).reset_index(drop=True)
#print(data)

def pca(input_file, dim):
    input_file = input_file.drop(["category_id"], axis=1)
    #scale data
    scaled_data = StandardScaler().fit_transform(input_file)
    features = scaled_data.T
    #print(features)
    
    cov_matrix = np.cov(features)
    #print(cov_matrix)
    values, vectors = np.linalg.eig(cov_matrix)
    
    #calculate the percentage of explained variance per principal component
    explained_variances = []
    for i in range(len(values)):
        explained_variances.append(values[i] / np.sum(values))
 
    #print(np.sum(explained_variances), '\n', explained_variances, '\n')
    
    projected_1 = scaled_data.dot(vectors.T[0])
    projected_2 = scaled_data.dot(vectors.T[1])
    
    res = pd.DataFrame(projected_1, columns=['PC1'])
    res['PC2'] = projected_2
    #print(res.head())
    """
    vectors = scaled_data.dot(vectors.T)
    
    res = pd.DataFrame(columns=["PC1"])
    
    for i in range(dim):
        temp_res = pd.DataFrame(vectors[i], columns=["PC{}".format(i+1)])
        res = pd.concat([res, temp_res])
    """
    return res
    
result = pca(data, 2)
print(result.head())
plt.figure(figsize=(20, 10))
sns.scatterplot(data=result, x="PC1", y="PC2", hue=data["category_id"])