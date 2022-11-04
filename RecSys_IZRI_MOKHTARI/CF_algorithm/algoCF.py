import numpy as np
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")



class Collaborative_filtering_item_item():
    """
        Class that allows us to compute the signature matrix, apply LSH and run collaborative filtering algorithm
    """

    def __init__(self, utility_matrix_training, movies_rated_training_idx, similarity_matrix, 
                       n_hyperplanes=20, m_bands=5, k_sim_movies = 10, threshold=0.5):

        self.utility_matrix_training = utility_matrix_training          # utility matrix to use to make predictions
        self.movies_rated_training_idx = movies_rated_training_idx      # List of sets of movies rated by each user (list[0]: movies rated by user 1, ...)
        self.similarity_matrix = similarity_matrix                      # Similarity matrix between the movies
        self.n_hyperplanes = n_hyperplanes                              # Number of hyperplanes to construct the signature matrix
        self.m_bands = m_bands                                          # Number of bands we will split into our sig_matrix. 
        self.k_sim_movies = k_sim_movies                                # Number of similar items to take into account 
        self.threshold = threshold                                      # Threshold: threshold to concider similar movies

    def predictions(self):
        """
            Main function:
                - Computes signature matrix of data and results of LSH
                - Run CF algorithm and return its results 
        """
        ########### Compute signature matrix
        print('-- Compute signature matrix  ...')
        sig_matrix = self.signature_matrix()

        ########### Compute candidate movies similar
        print('Done.\n-- LSH  ...')
        self.candidate_movies = self.LSH(sig_matrix)

        ########### Learning ratings via collaborative filtering item item ###########
        print('Done.\n\n########### Learning ratings via collaborative filtering (item item view) ###########') 
        print('-- Run collaborative filtering algorithm ...\n (i.e Compute new utility matrix with ratings for all pairs (movies, users))')
        Utility_matrix_predicted, predictions_users = self.algorithmCF()

        return Utility_matrix_predicted, predictions_users





    ###############################################################################
    #                              Signature matrix                               # 
    ###############################################################################
    def signature_matrix(self):
        """
            Compute the signature matrix of the utility matrix.
        """
        # Dim of our movies
        dim_movies = self.utility_matrix_training.shape[0]

        # Generate normal vectors of hyperplanes 
        w = np.random.normal(0, 1, (self.n_hyperplanes, dim_movies) )

        # Signature matrix
        sig_matrix = (w @ np.array(self.utility_matrix_training) >=0).astype(int)

        return sig_matrix


    ###############################################################################
    #                              LSH Algorithm                                  # 
    ###############################################################################

    def LSH(self, sig_matrix):
        """
        LSH algorithm using the signature matrix, we group elements that are probably similar 
        into buckets.

        param:
            sig_matrix (2D array)
        return: 
            dict[key] = values, where key is a movie i and value is all the movies potentially similar to i.
        """
        # Constants
        n_movies = sig_matrix.shape[1]
        size_bands = int(sig_matrix.shape[0]/self.m_bands)

        # dict for candidate movies 
        movies_candidates = {}

        for i in range(0, sig_matrix.shape[0], size_bands):
            band_dict = {}
            
            # For each movie
            for movie in range(n_movies):
                # Retreive partial signature
                partial_sig = sig_matrix[i:i+size_bands, movie]
                hash_partial_sig = '_'.join(partial_sig.astype(str))

                # Add the movie to the dict
                if hash_partial_sig not in band_dict:
                    band_dict[hash_partial_sig] = [movie]
                else:
                    band_dict[hash_partial_sig].append(movie)
            # We save all candidate movies of a bucket in a dict
            # Ex: if we have a bucket with these movies [1, 2, 3]:
            #       we will save have: d[1]=[2,3], d[2] = [1, 3], d[3] = [1, 2]
            for list_movies in band_dict.values():
                for movie in list_movies:
                    if not movie in movies_candidates:
                        movies_candidates[movie] = set(list_movies) - {movie}
                    else:
                        movies_candidates[movie].update(set(list_movies) - {movie})

        return movies_candidates


    ###############################################################################
    #                 Collaborative filtering algorithm                           # 
    ###############################################################################

    def algorithmCF(self):
        """
        Recommendation algorithm: computes predictions ratings for each user and movie using CF algorithm and item item view
        return:
            Utility_matrix_predictions: Utility matrix filled with the predictions
            predictions_user (dict):  predictions for each user 
                                    (key:user id, value:list of pairs (movie, rating))
        """
        # Copy the utlity matrix 
        r = self.utility_matrix_training.copy()
        r[r==-1] = 0

        # Average of ratings
        mu = r[r!=-1].mean()
        b_users = r.sum(1)/(self.utility_matrix_training!=-1).sum(axis=1) - mu 
        b_movies = r.sum(0)/(self.utility_matrix_training!=-1).sum(axis=0) - mu


        # Utility matrix we will fill with predictions
        Utility_Matrix_predictions = np.zeros(r.shape)

        # Dictionnary with predictions sorted for each user (key:user id, value:list of pairs (movie, rating))
        predictions_users = {}

        # Compute ratings for all users
        for user_idx in tqdm(range(r.shape[0])):
            # Rating deviation of user
            bx = b_users[user_idx]

            # list of predictions for user
            movies_ratings_predicted = []

            for movie_i_idx in range(r.shape[1]):
                # Rating deviation of movie i
                bi = b_movies[movie_i_idx]

                # Handle movies not rated 
                bi = 0 if np.isnan(bi) else bi
        
                # Baseline estimator i
                bxi = mu + bi + bx 

                # most similar movies of movie "i" and already seen by the user
                similar_movies_of_user = self.k_similar_movies(movie_i_idx, user_idx)

                # Compute predicted rating
                rxi = 0
                if similar_movies_of_user != []:
                    sum_total = 0
                    for (movie_j_idx, sim_j) in similar_movies_of_user:
                        # Rating deviation of movie j
                        bj = b_movies[movie_j_idx]
                        # Baseline estimator j
                        bxj = mu + bj + bx
                        rxi += sim_j * (r[user_idx][movie_j_idx] - bxj)
                        sum_total += sim_j
                    # Predicted rating for movie i
                    rxi = bxi + (rxi/sum_total)

                # Bound rxi between 0 and 5
                rxi = 5. if rxi > 5 else 0. if rxi <= 0. else rxi

                # Fill utility matrix
                Utility_Matrix_predictions[user_idx][movie_i_idx] = rxi
                movies_ratings_predicted.append((movie_i_idx, rxi))
            movies_ratings_predicted.sort(key = lambda x: x[1], reverse=True)
            predictions_users[user_idx + 1] = movies_ratings_predicted

        return Utility_Matrix_predictions, predictions_users 
   
    
    ###############################################################################
    #                 Similar movies given a movie and an user                    # 
    ###############################################################################
    def k_similar_movies(self, movie_i_idx, user_idx):
        """
            Given a movie i, we return similar movies to it already seen by the user given.

            Params:
            movie_i_idx: index of movie in Utility matrix
            user_idx: index of user un Utility_matrix
            Return:
            list (movie, similarity) of most similar movies

        """
        # We care only about items seen by the user and similar to the one given  
        movies_to_check = self.candidate_movies[movie_i_idx].intersection(self.movies_rated_training_idx[user_idx])
        # Keep items above threshold
        movies_similarities_pairs = [(movie_j_idx, self.similarity_matrix[movie_i_idx, movie_j_idx]) \
                                        for movie_j_idx in movies_to_check \
                                        if self.similarity_matrix[movie_j_idx][movie_i_idx] > self.threshold]
        # Sort and return the most similar
        movies_similarities_pairs.sort(key = lambda x: x[1], reverse=True) 
        most_sim = movies_similarities_pairs[:self.k_sim_movies]
        return most_sim

