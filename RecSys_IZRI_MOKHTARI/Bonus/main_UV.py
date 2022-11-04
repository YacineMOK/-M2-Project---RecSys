import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from UV_decomposition import Config_A, Config_B
import time


PATH = "../ml-latest-small/"
FILES = ["links", "movies", "ratings", "tags"]

df_dict = {}
for filename in FILES:
    df_dict[filename] = pd.read_csv(PATH+filename+".csv", sep=",")

# List (user_id) and (list movie_ids)
num_max_users = max(np.max(df_dict["ratings"]["userId"].values), np.max(df_dict["tags"]["userId"].values))
users_Ids = np.arange(num_max_users)
movies_Ids = np.sort(df_dict["movies"]["movieId"].values)

# defining useful conversion functions
def movieId2Index(mId):
    """
    From movieId to movieIndex
    Arg:
      - mId : Movie Id array
    Returns:
      - the corresponding movie index array in the Utility Matrix
    """
    return [np.argwhere(np.array(movies_Ids) == i)[0,0] for i in mId]

def movieIndex2Id(mInd):
    """
    From movieIndex to movieId
    Arg:
      - mInd : Array of movie indeces in the Utility Matrix
    Returns:
      - The corresponding movie Id array in the MovieLens db
    """
    return [movies_Ids[i] for i in mInd]


# list couple (user x movies_rated) , list (ratings)
userXmovie_list, rating_list = df_dict['ratings'][['userId', 'movieId']].values  , df_dict['ratings']['rating'].values

# function to split the 

# build matrix
def build_matrix_from_set(test_size=0.01, verbose=True):
    # split
    X_train, X_test, y_train, y_test = train_test_split(userXmovie_list, rating_list, test_size=test_size, random_state=42)

    # build
    Utility_Matrix = np.empty((len(users_Ids), len(movies_Ids)))
    Utility_Matrix[:] = np.nan

    for i in range(X_train.shape[0]):
        (user, movie) = X_train[i]
        Utility_Matrix[user-1, movieId2Index([movie])] = y_train[i]

    if verbose:
        print("Number of Nan in the Utility Matrix: ",np.isnan(Utility_Matrix).sum()," | Train set size: ", X_train.shape[0], " | Test set size: ", X_test.shape[0])
    
    return Utility_Matrix, X_train, X_test, y_train, y_test

def  alreadyWatched(user):
    res = [movieId2Index([t[1]]) for t in userXmovie_list if t[0] == user]
    return res

if __name__ == "__main__":
    MaxIter =200
    k = 2
    print("Welcome, my name is RecSys:\nPlease wait a little bit before querying...")
    Utility_Matrix, X_train , X_test, _, y_test = build_matrix_from_set(test_size = 0.1)
    B_model = Config_B(Utility_Matrix, latent_size = [k]*3, MaxIter=[MaxIter]*3)
    B_model.opt()
    B_M_hat = B_model.getM_hat()
    print("We are ready :D !")
    user_id = 0
    while True:
      while user_id <= 0 or user_id > 610:
        user_id = int(input("Which user are you? (Insert User id from 1 to 610) "))
      print("Welcome back user"+str(user_id))
      k = int(input("How many recommendation do you want?"))
      deja_vu = alreadyWatched(user_id)
      print(len(deja_vu))
      mask = np.array([False]*B_M_hat.shape[1])
      mask[deja_vu] = True
      res = B_model.recommend_k_item(user_id-1, k=k, mask = mask)
      print("Here are your recommendation:")
      for i in range(len(res)):
        print(str(i)+". " + str(df_dict["movies"].loc[df_dict["movies"]['movieId'] == movieIndex2Id([res[i]])[0]]['title'].values[0]))

    