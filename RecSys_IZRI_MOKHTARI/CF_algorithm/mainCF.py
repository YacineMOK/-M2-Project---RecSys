import numpy as np

from algoCF import *
from dataHandler import *


PATH = "../ml-latest-small/"        # /!\ Path to data files /!\




###############################################################################
#                              Main program                                   # 
###############################################################################


def main():
    # Fix the seed to repeat tests
    np.random.seed(0)

    print('########### Loading and preparing data ###########') 

    # Loading and splitting data
    data = DataHandler(PATH)
    data.load_data()
    test_size = 0.1
    X_train, X_test, y_train, y_test = data.split_dataset(test_size)

    # Build our initial utility matrix
    print('-- Build initial utility matrix ...')
    utility_matrix_training = data.build_utility_matrix(X_train, y_train)
    print('Done.')

    # CF algorithm on training data
    n_hyperplanes = 30
    m_bands = 3
    k_sim_movies = 10
    threshold = 0.4
    cf = Collaborative_filtering_item_item(utility_matrix_training, data.movies_rated_training_idx, data.similarity_matrix,\
                                           n_hyperplanes, m_bands, k_sim_movies, threshold)


    ########### Collaborative filtering item item predictions ###########
    Utility_matrix_predicted, predictions_users = cf.predictions()

    # data.print_rmse(Utility_matrix_predicted, X_test, y_test, 'test')
    print('\n########### Recommendation System ###########') 
    while (True):
        print("\n- Ready :) Please enter an user id (between 1 and 610):")
        print ("-- Or type 'exit' to quit. --\n")
        choice = input()
        if (choice == 'exit'):
            exit()
        else:
            try:
                # Print recommendations given user id and k
                user_id  = int(choice)
                while (user_id < 1 or user_id > 610):
                    print("\n\n- Please enter an user id (between 1 and 610):")
                    user_id  = input()

                print("- Please enter a number of movies to recommend:")
                k  = int(input())
                data.print_movies_recommended(predictions_users, user_id, k)  
            except:
                continue

        print('##################################')


if __name__ == '__main__':
    main()
