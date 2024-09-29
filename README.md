# Som on disney movies
# Self-Organizing Map (SOM) Implementation for Disney Movie Gross Analysis
Dataset:

Disney Movie Total Gross, which includes the following features:

Movie Title: The title of the movie.

Date Released: The release date of the movie.

Genre: The genre(s) of the movie.

MPAA Rating: The rating assigned to the movie (e.g., G, PG, PG-13, R).

Total Gross: The total gross earnings of the movie.

Inflation Adjusted Gross: The total gross adjusted for inflation.

The dataset is in CSV format and can be downloaded from here.

# Software Requirements
Google colab

python

# CODE
!pip install minisom
!pip install pandas
!pip install numpy
!pip install matplotlib
from google.colab import files
uploaded = files.upload()
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv('disney movie total gross.csv')
print(data.head())
print(data.dtypes)
numeric_features = data[['Total Gross', 'Inflation Adjusted Gross']]
data_encoded = pd.get_dummies(data, columns=['Genre', 'MPAA Rating'], drop_first=True)
numeric_features_encoded = data_encoded[['Total Gross', 'Inflation Adjusted Gross']]
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(numeric_features_encoded)
normalized_df = pd.DataFrame(data_normalized, columns=numeric_features_encoded.columns)
print(normalized_df.head())
from minisom import MiniSom
import numpy as np
som = MiniSom(24, 24, data_normalized.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(data_normalized)
som.train_random(data_normalized, 1000)
from pylab import bone, pcolor, colorbar, plot
bone()
pcolor(som.distance_map().T)
colorbar()
