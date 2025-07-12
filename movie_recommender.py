import pandas as pd
import numpy as np
import sys
from flask import Flask, request, render_template_string

#Load the data
ratings = pd.read_csv("ml-100k/u.data",sep='\t',header=None,names=
["user_id", "movie_id", "rating", "timestamp"])

movies = pd.read_csv("ml-100k/u.item", sep='|', header=None, encoding='latin-1',

names=["movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
"unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
"Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
"Romance", "Sci-Fi", "Thriller", "War", "Western"])


#Merge both datasets on movie_id since that's the common column/row values between each one
merged = pd.merge(ratings,movies,on='movie_id')

#Drop any unneccesary columns
merged = merged.drop(['timestamp','release_date','video_release_date','IMDb_URL'],axis=1)


#Create a matrix which is useful for implementing recommendation algorithms
#merged is the data it's based off of, values will be audience reviews, index will be row values, columns will be column values
user_movie_matrix = pd.pivot_table(data=merged, values='rating',index='user_id',columns='title')



#Calculates cosine similarity between 2 vectors
#1 means most similar, -1 most dissimilar, 0 neutral
def cosine_similarity(v1,v2):
    dot_product = np.dot(v1,v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    
    return (dot_product)/(magnitude_v1 * magnitude_v2)




#function to recommend top N similar movies
def recommend_movies(movie_title, ratings_matrix, top_n):
    #if user movie input not in matrix, return error message
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
        #get rid of Na rating values
        common_ratings = pd.concat([movie_column_vector,movie_rating],axis=1).dropna()
        
        #compute similarity between user movie input column vector and every other movie title vector
        if not common_ratings.empty and len(common_ratings)>=20:
            similarity = cosine_similarity(common_ratings.iloc[:,0],common_ratings.iloc[:,1])
            #store movie titles with similarity value in dictionary
            similarity_dict[col] = similarity
        #print(f"{col}: {len(common_ratings)} users in common")

    
    
    #sort dictionary in decreasing order
    sorted_dict = dict(sorted(similarity_dict.items(),key = lambda x: x[1],reverse=True))

    #list of top_n movie titles
    movie_lst = []
    #Loop through sorted dictionary
    for title in sorted_dict:
        #append top_n movies to the list
        movie_lst.append(title)
        if len(movie_lst) == top_n:
            break
    
    return movie_lst


# Check if at least one argument is provided
# if len(sys.argv) > 1:
#     input_arg = sys.argv[1]  # First command-line argument
#     print(f"Input argument: {input_arg}")
#     print(recommend_movies(input_arg,user_movie_matrix,5))
# else:
#     print("No command-line argument provided.")

print(recommend_movies("Pulp Fiction (1994)",user_movie_matrix,5))
#print(recommend_movies("Star Wars (1977)",user_movie_matrix,5))
#print(recommend_movies("Jurassic Park (1993)",user_movie_matrix,5))



### Run Flask Server for Web GUI

# app = Flask(__name__)

# # HTML template for the input form and output display
# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html>
# <head>
#     <title>Simple Input Processor</title>
#     <style>
#         body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
#         .container { text-align: center; }
#         input[type="text"] { padding: 8px; width: 200px; }
#         input[type="submit"] { padding: 8px 16px; margin-left: 10px; }
#         .output { margin-top: 20px; }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <h1>Input Processor</h1>
#         <form method="POST" action="/">
#             <input type="text" name="user_input" placeholder="Enter your input" required>
#             <input type="submit" value="Submit">
#         </form>
#         {% if output %}
#             <div class="output">
#                 <h3>Output:</h3>
#                 <p>{{ output }}</p>
#             </div>
#         {% endif %}
#     </div>
# </body>
# </html>
# """

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     output = None
#     if request.method == 'POST':
#         user_input = request.form.get('user_input')
#         try:
#             output = recommend_movies(user_input,user_movie_matrix,5)
            
#         except Exception as e:
#             output = user_input
#     return render_template_string(HTML_TEMPLATE, output=output)

# if __name__ == '__main__':
#     app.run(debug=True)