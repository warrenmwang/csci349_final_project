import pickle
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate, GridSearchCV

userID2userNameMap = {}
userName2userIDMap = {}

animeID2animeNameMap = {}
animeName2animeIDMap = {}

with open('userName2userIDMap.pkl', 'rb') as f:
    userName2userIDMap = pickle.load(f)

with open('userID2userNameMap.pkl', 'rb') as f:
    userID2userNameMap = pickle.load(f)


with open('animeID2animeNameMap.pkl', 'rb') as f:
    animeID2animeNameMap = pickle.load(f)

with open('animeName2animeIDMap.pkl', 'rb') as f:
    animeName2animeIDMap = pickle.load(f)

df = None
with open('df_custom.pkl', 'rb') as f:
    df = pickle.load(f)

user_ids = None
with open('user_ids.pkl', 'rb') as f:
    user_ids = pickle.load(f)

# Define a Reader object to parse the file
reader = Reader(rating_scale=(1, 10))  # Assuming scores are from 1 to 10 

# Load the dataset
data = Dataset.load_from_df(df[['user_id', 'anime_id', 'score']], reader)

# SVD
# https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD

# GridSearchCV
# https://surprise.readthedocs.io/en/stable/model_selection.html#surprise.model_selection.search.GridSearchCV
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

parameters = {
    'n_factors': [100, 200, 300],
    'n_epochs': [20, 30],
    'lr_all': [0.005, 0.001],
    'reg_all': [0.02, 0.03],
}

grid = GridSearchCV(algo_class = SVD, 
                   param_grid = parameters,
                   measures = ['rmse', 'mae'],
                   n_jobs = 2,
                   cv = 5,
                   refit = True,
                   joblib_verbose = 1)

print("Running GridSearch now...")
grid.fit(data)

with open('grid.pkl', 'wb') as f:
    pickle.dump(grid, f)
print("Saved grid.pkl")

def get_top_n_recommendations(algo, user_id, n=10):
    # Assume we have a list of all anime_ids in the dataset
    all_anime_ids = set(df['anime_id'].unique())
    
    # Get the list of anime_ids that the user has already rated
    rated_anime_ids = set(df[df['user_id'] == user_id]['anime_id'].unique())
    
    # Predict ratings for all anime the user hasn't rated
    predictions = [algo.predict(user_id, anime_id) for anime_id in all_anime_ids if anime_id not in rated_anime_ids]

    # print(predictions)
    
    # Sort the predictions in descending order of the estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Return the top N anime_ids
    top_n_anime_ids = [pred.iid for pred in predictions[:n]]
    return top_n_anime_ids


unique_user_ids = list(set(user_ids))

for test_user_id in unique_user_ids[:5]:
    top_recommendations = get_top_n_recommendations(algo=grid, user_id=str(test_user_id), n=10)
    print(f"Recommendations for {userID2userNameMap[test_user_id]}: ")
    for rec in top_recommendations:
        print(animeID2animeNameMap[rec])
    print()