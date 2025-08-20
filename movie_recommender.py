import pandas as pd
import numpy as np


#Load ratings data (user_id, movie_id, rating, timestamp)
ratings = pd.read_csv("ml-100k/u.data",sep='\t',header=None,names=
["user_id", "movie_id", "rating", "timestamp"])

#load movies data (movie details + genres)
movies = pd.read_csv("ml-100k/u.item", sep='|', header=None, encoding='latin-1',names=["movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
"unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
"Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
"Romance", "Sci-Fi", "Thriller", "War", "Western"])

#list of genre columns
genre_cols = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
              "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
              "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

#combines all movie genres into a string
def combine_genres(row):
    return '|'.join([g for g in genre_cols if row[g] == 1])


#add combined genres column to movies dataframe
movies['genres'] = movies.apply(combine_genres, axis=1)

#Merge both datasets (movies + ratings) on movie_id since that's the common column/row values between each one
merged = pd.merge(ratings,movies,on='movie_id')

#Drop any unneccesary columns
merged = merged.drop(['timestamp','release_date','video_release_date','IMDb_URL'],axis=1)


#Create a matrix which is useful for implementing recommendation algorithms
#merged is the data it's based off of, values will be audience reviews, index will be row values, columns will be column values
user_movie_matrix = pd.pivot_table(data=merged, values='rating',index='user_id',columns='title')



#Calculates cosine similarity between 2 vectors
#1 means most similar, -1 most dissimilar, 0 neutral
def cosine_similarity(v1,v2):
    dot_product = np.dot(v1,v2) #dot product between vectors
    magnitude_v1 = np.linalg.norm(v1) #length of vector 1
    magnitude_v2 = np.linalg.norm(v2) #length of vector 2

    if magnitude_v1 == 0 or magnitude_v2 == 0: #avoid division by 0
        return 0
    
    return (dot_product)/(magnitude_v1 * magnitude_v2) #cosine similarity formula


#function to recommend top N similar movies to a given movie
def recommend_movies(movie_title, ratings_matrix, top_n):

    #get genres of input movie
    genres = movies.loc[movies['title'] == movie_title,'genres'].iloc[0]
    genres_set = set(genres.split('|'))

    #if input movie not in dataset, return error message
    if movie_title not in ratings_matrix.columns:
        return "Movie not found in data"

    #extract column vector of user movie input
    movie_column_vector = ratings_matrix[movie_title]

    #every column in the rating matrix except user movie input column
    columns_to_loop = ratings_matrix.columns.difference([movie_title])

    #empty dictionary to store similarity values and coressponding movie title
    similarity_dict = {}

    #loop through ratings_matrix
    for col in columns_to_loop:
        movie_rating = ratings_matrix[col]

        #only keep users who rated both movies
        common_ratings = pd.concat([movie_column_vector,movie_rating],axis=1).dropna()

        #get genres of other movie
        other_genres = movies.loc[movies['title']==col,'genres'].iloc[0]
        other_genres_set = set(other_genres.split('|'))
        
        #only compare if both movies share at least 1 genre
        if genres_set.intersection(other_genres_set):

            #only compare similarity if enough users rated both
            if not common_ratings.empty and len(common_ratings)>=20:
                similarity = cosine_similarity(common_ratings.iloc[:,0],common_ratings.iloc[:,1])

                #store movie titles with similarity value in dictionary
                similarity_dict[col] = similarity


    
    
    #sort dictionary in decreasing order
    sorted_dict = dict(sorted(similarity_dict.items(),key = lambda x: x[1],reverse=True))

    #list of top N most similar movies
    movie_lst = []

    #Loop through sorted dictionary
    for title in sorted_dict:
        #append top_n movies to the list
        movie_lst.append(title)
        if len(movie_lst) == top_n: #stop when enough recommendations are found
            break
    
    return movie_lst



