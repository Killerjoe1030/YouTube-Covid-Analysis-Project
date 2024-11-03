import numpy as np 
import pandas as pd 
import sys
import matplotlib.pyplot as plt 
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def proportion_test(test_col, data_path):
    # indices:
    # 0 - 2016
    # 1 - 2017
    # 2 - 2018 
    # 3 - 2019 
    # 4 - 2020 
    # 5 - 2021 
    # 6 - 2022
    
    print("\n-----------------" + test_col + "-----------------")
    data_years = []
    for year in range(2016,2023):
        data_years.append(pd.read_json(data_path + str(year) + ".json"))    

    # filter viewcounts and remove video rows where test_col == 0
    # use sqrt to transform data to normalise
    plt.figure(figsize=(10,5))
    for i in range(7):
        data_years[i] = data_years[i][(data_years[i][test_col] > 0) & (data_years[i]["Views"] < 80000000) & (data_years[i]["Views"] > 5000)].copy()
        data_years[i][test_col + "_ratio"] = np.sqrt(data_years[i][test_col] / data_years[i]["Views"])
        
        # normal test
        print("Normal test " + str(2016 + i) + ": " , stats.normaltest(data_years[i][test_col + "_ratio"]).pvalue)  
        plt.subplot(2,4,i + 1)
        plt.hist(data_years[i][test_col + "_ratio"])    
        plt.title(str(2016 + i) + " " + test_col + ' ratio')
    
    plt.tight_layout()
    plt.savefig("figures/" + test_col.lower() + "_ratio_normaltest.png")
    print("\n")

    # not consistent?
    print("T-test: ", stats.ttest_ind(data_years[3][test_col + "_ratio"],data_years[4][test_col + "_ratio"]).pvalue)
    print("Mann-Whitney U test: ", stats.mannwhitneyu(data_years[3][test_col + "_ratio"], data_years[4][test_col + "_ratio"]).pvalue, "\n")

    # ANOVA: check for significance
    print("ANOVA: " + str(stats.f_oneway(*[data_years[col][test_col + "_ratio"] for col in range(7)]).pvalue))    

    # concat melted data for posthoc
    melted_data = pd.concat([*[data_years[col] for col in range(7)]])
    posthoc = pairwise_tukeyhsd(melted_data[test_col + "_ratio"],melted_data['Date'], alpha=0.05)

    # check table and plot
    print(posthoc)
    posthoc.plot_simultaneous()
    plt.savefig("figures/" + test_col.lower() + "_tukey.png")
    
    return



def main(data_path):

    # Likes: Did the quality of the vids stay consistent or did change after covid? 
    proportion_test('Likes', data_path)
    
    # Comments: Were the users more/less engaging on the platform after covid?
    proportion_test('Comments', data_path)


if __name__  == "__main__":
    prefix = sys.argv[1]
    main("@youtube_api_get/" + prefix + "_youtube_stats_" )