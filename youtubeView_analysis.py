import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sys
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats

import warnings 
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

def view_filter(yt_data):
    # look for vids with viewcount > 500 and < 80mil, otherwise it will be too right skewed to transform
    yt_data_fil = yt_data[(yt_data["Views"] > 5000) & (yt_data["Views"] < 80000000)].copy()
    
    # transform data since it is right skewed
    # pareto distribution so log will work best
    yt_data_fil['log_views'] = np.log(yt_data_fil['Views'])
    return yt_data_fil[['Date','log_views']]


def main(data_path):
    data_years = []
    
    # indices:
    # 0 - 2016
    # 1 - 2017
    # 2 - 2018 
    # 3 - 2019 
    # 4 - 2020 
    # 5 - 2021 
    # 6 - 2022

    # Read the data in
    for year in range(2016,2023):
        data_years.append(pd.read_json(data_path + str(year) + ".json"))

    # for mannwhitney
    views_2019 = data_years[3]
    views_2020 = data_years[4]

    # filter by views to help with normalising the data
    views_2019 = views_2019[(views_2019["Views"] < 80000000) & (views_2019["Views"] > 5000)].copy()
    views_2020 = views_2020[(views_2020["Views"] < 80000000) & (views_2020["Views"] > 5000)].copy()

    # transform all years using log (since its right skewed) and filtering view counts
    transformed_data = []
    for i in range(7):
        transformed_data.append(view_filter(data_years[i]))

    # concat all years into one df
    melted_data = pd.concat([*[transformed_data[col] for col in range(7)]])

    # check normal test (probbably wont pass, but should be relatively normal by CLT)
    # plot normal tests
    plt.figure(figsize=(10,5))
    for i in range(7):
        print("Normal test " + str(2016 + i) + ": " , stats.normaltest(transformed_data[i]["log_views"]).pvalue)  
        plt.subplot(2,4,i + 1)
        plt.hist(transformed_data[i]["log_views"])    
        plt.title(str(2016 + i) + ' views')
    
    plt.tight_layout()
    plt.savefig("figures/views_normaltest.png")
    print("\n")

    # T-test and mannwhitneyu: to check if means were different in 2019 and 2020
    ttest = stats.ttest_ind(transformed_data[3]["log_views"], transformed_data[4]["log_views"])
    print("T-test (pre vs post covid): ", ttest.pvalue)
    print("Mann-Whitney U test: ", stats.mannwhitneyu(views_2019["Views"], views_2020["Views"]).pvalue, "\n")

    # ANOVA: see if test is significant
    # data looks normal when graphed
    print("ANOVA: ", stats.f_oneway(*[transformed_data[col] for col in range(7)]).pvalue)

    # Pairwise Tukey: posthoc test to see if any significance between pre and post covid
    posthoc = pairwise_tukeyhsd(melted_data['log_views'],melted_data['Date'], alpha=0.05)

    # analyse table and plot to see pattern
    print(posthoc)
    posthoc.plot_simultaneous()
    plt.savefig("figures/views_tukey.png")


if __name__  == "__main__":
    prefix = sys.argv[1]
    main("@youtube_api_get/" + prefix + "_youtube_stats_" )