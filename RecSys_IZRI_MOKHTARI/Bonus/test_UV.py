import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from UV_decomposition import Config_A, Config_B
import time


PATH = "./ml-latest-small/"
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
    


# appel fonction
def rmse_test(M_hat, X_test, y_test):
    # first estimator
    mse = 0
    for i in range(X_test.shape[0]):
        (user, movie) = X_test[i]
        est = M_hat[user-1, movieId2Index([movie])] 
        mse += (est - y_test[i])**2
    mse /= X_test.shape[0]
    return np.sqrt(mse)

if __name__ == "__main__":
    print("Welcome to the testing program... :::")

    f = open("results.csv", 'w')
    plot = open('plot_training_loss.npy', 'w')

    f.write("latent size (k),test_size,MaxIter,opt time,rmse (on the test set)\n")

    test_sizes = [0.1]
    latent_sizes = [2, 5, 15, 50]
    iterations = [20, 200, 500, 1000]

    for k in latent_sizes:
        print("--K = ",k)
        for test_size in test_sizes:
            print("--test_size = ", test_size)
            for MaxIter in iterations:
                print("--MaxIter = ", MaxIter)
                f.write(str(k)+",")
                f.write(str(test_size)+",")
                f.write(str(MaxIter)+",")

                # utility matrix
                Utility_Matrix, _, X_test, _, y_test = build_matrix_from_set(test_size = test_size)
                A_model = Config_A(Utility_Matrix, latent_size = k, test_size=test_size, MaxIter=MaxIter)
                
                # Optimization
                print("\t--Begin opt")
                start = time.time()
                A_model.opt()
                end = time.time()
                f.write(str(np.round(end-start, 3))+ "s,")

                # for the plot 
                A_model.save_rmse_tracking()

                # rmse
                A_M_hat = A_model.getM_hat()
                rmse = rmse_test(A_M_hat, X_test, y_test)
                print("`\t--RMSE (test): ", rmse)
                f.write(str(np.round(rmse, 3))+ "\n")
        print("- "*10)

    