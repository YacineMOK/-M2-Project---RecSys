import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


FILES = ["links", "movies", "ratings", "tags"]




class DataHandler():
    """
        Class to load data, split it, compute utility matrix and some useful functions 
    """
    def __init__(self, path):
        self.path = path                        # Path of the folder to data   
        self.df_data = None                     # Dict that will contain all our initial data      
        self.users_Ids = None                   # All User ids        
        self.movies_rated = None                # Ids of mvies rated by each user (list[0]: movies rated by user of id 1) 
        self.movies_rated_training_idx = None   # Indices of movies rated by each user (list[0]: movies rated by user of id 1) 
        self.movies_ratings = None              # Movies ratings for each user (list[0]: ratings by user of id 1)
        self.movies_Ids = None                  # All Movie ids


          
    ###############################################################################
    #                Loading, splitting data and useful functions                 # 
    ###############################################################################
    def load_data(self):
        # Load all data in a dict
        df_data = {}
        for filename in FILES:
            df_data[filename] = pd.read_csv(self.path + filename + ".csv", sep=",")
        self.df_data = df_data
        self.get_info_movies_users()


    def get_info_movies_users(self):
        # Get ids of movies and users
        num_max_users = max(np.max(self.df_data["ratings"]["userId"].values), np.max(self.df_data["tags"]["userId"].values))

        self.users_Ids = np.arange(num_max_users)
        self.movies_Ids = np.sort(self.df_data["movies"]["movieId"].values)

        # Get movies rated by users and their ratings
        # to do that, we use the groupby function of pandas
        grouped_ratings = self.df_data["ratings"].groupby('userId')

        # Ids of movies rated
        self.movies_rated = grouped_ratings['movieId'].apply(list).reset_index(name='Movies rated')["Movies rated"].values
        # The ratings
        self.movies_ratings = grouped_ratings['rating'].apply(list).reset_index(name='Movies ratings')["Movies ratings"].values


    def movieId2Index(self, mId):
        """
        From movieId to movieIndex
        Arg:
        - mId : Movie Id array
        Returns:
        - the corresponding movie index array in the Utility Matrix
        """
        return [np.argwhere(np.array(self.movies_Ids) == i)[0,0] for i in mId]


    def movieIndex2Id(self, mInd):
        """
        From movieIndex to movieId
        Arg:
        - mInd : Array of movie indeces in the Utility Matrix
        Returns:
        - The corresponding movie Id array in the MovieLens db
        """
        return [self.movies_Ids[i] for i in mInd]


    def split_dataset(self, test_size):
        userXmovie_list, rating_list = self.df_data['ratings'][['userId', 'movieId']].values, self.df_data['ratings']['rating'].values
        X_train, X_test, y_train, y_test = train_test_split(userXmovie_list, rating_list, test_size=test_size)

        # Indices of the movies
        self.movies_rated_training_idx = {}
        for (user, movie) in X_train:
            if (user-1) not in self.movies_rated_training_idx:
                self.movies_rated_training_idx[user-1] = [self.movieId2Index([movie])[0]]
            else:
                self.movies_rated_training_idx[user-1].append(self.movieId2Index([movie])[0])

        return X_train, X_test, y_train, y_test



    ###############################################################################
    #                    Building utility and similarity matrix                   # 
    ###############################################################################


    def build_utility_matrix(self, X_train, y_train):
        """
            Build the utility matrix using training set
        """

        Utility_Matrix = -1 * np.ones((len(self.users_Ids), len(self.movies_Ids)))

        for i in range(X_train.shape[0]):
            (user, movie) = X_train[i]
            Utility_Matrix[user-1, self.movieId2Index([movie])] = y_train[i]
        self.similarity_matrix = self.build_similarity_matrix(Utility_Matrix)
        return Utility_Matrix


    def build_similarity_matrix(self, utility_matrix):
        """
            Build cosine similarity given the utility matrix
        """
        utility_matrix_transposed = (utility_matrix.T).copy()
        utility_matrix_transposed[utility_matrix_transposed==-1] = 0

        similarity_matrix = cosine_similarity(utility_matrix_transposed)

        return similarity_matrix



    ###############################################################################
    #             Functions to display predictions & evaluation                   # 
    ###############################################################################

    def print_movies_recommended(self, predictions, user_id, k=10):
        """
            Return k recommended movies for the user given
        """
        ## Movies already rated (to compare)
        print(f'##### Some movies already rated by user {user_id}#####')
        df_movies_rated = self.df_data["movies"].loc[self.df_data["movies"]['movieId'].isin(self.movies_rated[user_id-1])]
        df_movies_rated.insert(1, 'rating', self.movies_ratings[user_id-1])
        df_movies_rated = df_movies_rated.sort_values('rating', ascending=False).head(k)

        for i, (_, movie) in enumerate(df_movies_rated.iterrows(), start=1):
            m = movie['title']
            rating = movie['rating']
            genres = movie['genres']
            print(f"- {'Movie' + str(i) + ': ' + str(m) + '.':<75}  ({rating}/5) \t {genres + '.' :<30}")

        ## Movies recommended
        print(f'\n ##### {k} recommendations for user {user_id} #####')
        preds_user = predictions[user_id]
        counter = 0 
        for i, (movie, rating) in enumerate(preds_user, start=1):
            if counter >= k:
                break
            movie = self.df_data["movies"].loc[self.df_data["movies"]['movieId'] == self.movieIndex2Id([movie])[0]]
            # Recommend only movies that are not rated by the user in the initial dataset
            if movie['movieId'].values[0] not in self.movies_rated[user_id-1]:
                movie_title = str(movie['title'].values[0])
                movie_genres = movie['genres'].values[0]
                counter += 1
                print(f"- {'Movie ' + str(counter) + ': ' + movie_title + '.' :<75} ({np.round(rating, 2)}/5) \t {movie_genres  + '.' :<30}")
        print('\n')
    
    def print_rmse(self, Utility_matrix_predicted, X, y, name_set):
        """
            Computes and print the rmse for the given set X. y is the vector of true ratings.
        """
        mse = 0
        for i in range(X.shape[0]):
            (user, movie) = X[i]
            mse += (Utility_matrix_predicted[user-1][self.movieId2Index([movie])[0]] - y[i])**2

        mse /= X.shape[0]
        print(f"--- The RMSE value for {name_set} set is: {mse**0.5}") 


