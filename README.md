# Project 353: Youtube analysis on Covid-19 
built on: macOS Sonoma 14.0

## Required Tools
Python 3

### Libraries used for data management
* Pandas
* Numpy

### Libraries used for analysis
* Matplotlib
* SciPy
* Statsmodels
* Scikitlearn

### Libraries used for wrangling data
* [Google APIs Client Library for Python](https://developers.google.com/youtube/v3/quickstart/python)
* Datetime


## Running the .py files (in zsh terminal)
### All analysis files:
> python3 [filenamehere].py [prefix_of_json (re,re2,...)]

### Youtube API get (API keys will be included in the report)
> python3 @youtube_api_get/ytdata.py [API_KEY]


## Additional notes
* output figures and tables can be found in their corresponding directories ([/figures](/figures), [/tables](/tables), [/@youtube_api_get](/@youtube_api_get))
* [ytdata.py](@youtube_api_get/ytdata.py): year output can be modified in the 2nd line of main() function (line 72)
* [youtubeML.py](youtubeML.py): different sets of features in X can be modified to test accuracy (line 95-98)
* if needed, you can create an API key with a Google account [HERE](https://developers.google.com/youtube/v3)




