import pandas as pd 
import sys
import matplotlib.pyplot as plt 
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import make_pipeline

def get_clusters(X):
    model = make_pipeline(
        #AgglomerativeClustering(n_clusters=2)
        MiniBatchKMeans(n_clusters=2, n_init=10) # warning if n_init not set
    )
    model.fit(X)
    return model.fit_predict(X)


def main(data_path):

    data_years = []
    for year in range(2016,2023):
        data_years.append(pd.read_json(data_path + str(year) + ".json"))

    # data is too condesnse to make a decent cluster isolation
    for i in range(7):
        data_years[i] = data_years[i][(data_years[i]["Likes"] > 0) & (data_years[i]["Comments"] > 0) & (data_years[i]["Views"] < 2000000) & (data_years[i]["Views"] > 50000)].copy()
        data_years[i]['Covid'] = True if data_years[i]['Date'].any() > 2019 else False
        print(i + 2016, data_years[i].shape)

    all_data = pd.concat([*[data_years[col] for col in range(7)]])  
    X = all_data[['Likes','Views']]
    y = all_data['Covid']
    clusters = get_clusters(X)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X['Likes'], X['Views'], c=clusters, cmap='Set1', edgecolor='k', s=30)
    plt.savefig('figures/clusters.png')

    df = pd.DataFrame({
        'cluster': clusters,
        'Covid?': y,
    })
    
    # check proportion of each cluster
    counts = pd.crosstab(df['Covid?'], df['cluster'])
    print(counts)


if __name__  == "__main__":
    prefix = sys.argv[1]
    main("@youtube_api_get/" + prefix + "_youtube_stats_" )