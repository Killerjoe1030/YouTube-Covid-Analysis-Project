import pandas as pd 
import matplotlib.pyplot as plt 
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor  
from sklearn.neighbors import KNeighborsClassifier

def filter_noTags(yt_data):
    yt_data = yt_data[yt_data['Tags'].apply(lambda x: len(x) > 0)].copy()
    yt_data['covid'] = True if yt_data['Date'][0] > 2019 else False
    return yt_data


def filter_likes(yt_data):
    yt_data = yt_data[yt_data["Likes"] > 0 ].copy()
    return yt_data


def main(data_path):

    data_years = []
    for year in range(2016,2023):
        data_years.append(pd.read_json(data_path + str(year) + ".json"))

    filtered_data = []
    for i in range(7):
        filtered_data.append(filter_noTags(data_years[i]))

    grouped_data = []
    for i in range(7):
        filtered_data[i] = filtered_data[i].explode('Tags')
        filtered_data[i]['Tags'] = filtered_data[i]['Tags'].str.lower()#.apply(lambda word: word.lower())
        grouped_data.append(filtered_data[i].groupby('Tags').size().reset_index(name='count'))
        grouped_data[i] = grouped_data[i][grouped_data[i]['count'] > 1]

    tag_data = pd.concat([*[grouped_data[col] for col in range(7)]])


    # calculate the mean, median, and mode of the tag counts
    # mean = tag_data['count'].mean()
    # median = tag_data['count'].median()
    # mode = tag_data['count'].mode()
    # std = tag_data['count'].std()
    # min = tag_data['count'].min()
    # max = tag_data['count'].max()

    # Graphing a box plot of the tag counts 
    # plt.boxplot(tag_data['count'], vert=False)
    # plt.savefig("figures/tagCount_boxplot.png")

    tag_data.to_csv("tables/tag_counts.csv")
    plt.hist(tag_data['count'])
    plt.savefig("figures/tagCount_hist.png")

    ml_data = pd.concat([*[filtered_data[col] for col in range(7)]])     
    ml_data = ml_data[(ml_data['Likes'] > 0) & (ml_data['Comments'] > 0)].copy() # vids with 0 comments hurt accuracy 
    print(ml_data.shape)


    t = LabelEncoder()
    ml_data['Tags_encoded'] = t.fit_transform(ml_data['Tags'])

    X = ml_data
    y = ml_data[['Views','covid']]


    X_all, X_all2, y_train, y_valid = \
        train_test_split(X,y)
    

    X_train = X_all[['Views','Likes','Comments','Date','Tags_encoded']]
    X_valid = X_all2[['Views','Likes','Comments','Date','Tags_encoded']]


    view_model = make_pipeline(
        MinMaxScaler(),
        KNeighborsRegressor(10)        
    )

    # rank the features in terms of their independent contributions: 
    # lowest to highest: 
    # Date 
    # tags
    # comments
    # likes 

    # rank the features in terms of their contributions paired with comments: 
    # lowest to highest 
    # Tags 
    # Date
    # Likes

    # Replace the index column name for testing each features for X_train and X_valid
    view_model.fit(X_train[['Likes']],y_train['Views'])
    print('KNeighbors Regressor View Training Score: ' + str(view_model.score(X_train[['Likes']],y_train['Views'])))
    print('KNeighbors Regressor View Validation Score: ' + str(view_model.score(X_valid[['Likes']],y_valid['Views'])))
    predictions = view_model.predict(X_valid[['Likes']])

    correct_predictions = X_all2[y_valid['Views'] == predictions]
    wrong_predictions = X_all2[y_valid['Views'] != predictions]
    wrong_predictions.to_csv('tables/wrong_predictions.csv', index=False)
    correct_predictions.to_csv('tables/correct_predictions.csv', index=False)

    covid_model = make_pipeline(
        MinMaxScaler(),
        KNeighborsClassifier(n_neighbors=10)        
    ) 

    # consistetnly popular tags  have no correlation with covid 
    # lowers accuracy of model 
    # Views and likes are main contributors while comments is redundnant 
    # which is in line with our findings from the stats tests 
    covid_model.fit(X_train[['Views','Likes','Comments']],y_train['covid'])
    print('KNeighborsClassifier covid Training score: ' + str(covid_model.score(X_train[['Views','Likes','Comments']],y_train['covid'])))
    print('KNeighborsClassifier covid Validation Score: ' + str(covid_model.score(X_valid[['Views','Likes','Comments']],y_valid['covid'])))
    predictions_covid = covid_model.predict(X_valid[['Views','Likes','Comments']])
    correct_predictions2 = X_all2[y_valid['covid'] == predictions_covid]
    wrong_predictions2 = X_all2[y_valid['covid'] != predictions_covid]
    wrong_predictions2.to_csv('tables/wrong_predictions_covid.csv', index=False)
    correct_predictions2.to_csv('tables/correct_predictions_covid.csv', index=False)


if __name__  == "__main__":
    prefix = sys.argv[1]
    main("@youtube_api_get/" + prefix + "_youtube_stats_" )