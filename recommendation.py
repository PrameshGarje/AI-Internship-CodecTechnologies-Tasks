import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("movies.csv")

# Convert genres to matrix / features
cv = CountVectorizer()
genre_matrix = cv.fit_transform(df['genres'])

# Calculate cosine similarity between all movies
similarity = cosine_similarity(genre_matrix)

# Recommendation function
def recommend(movie):
    movie = movie.lower()
    
    if movie not in df['title'].str.lower().values:
        return ["Movie not found! Try another movie."]
    
    movie_index = df[df['title'].str.lower() == movie].index[0]
    distances = similarity[movie_index]

    movie_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]

    recommendations = []
    for i in movie_list:
        recommendations.append(df.iloc[i[0]].title)
    
    return recommendations

# User Input
movie_input = input("Enter a movie name: ")
result = recommend(movie_input)

print("\nRecommended Movies:")
for m in result:
    print("- " + m)
