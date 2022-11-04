import numpy as np

class UV_decomposition:
    def __init__(self, Utility_Matrix, latent_size=20, MaxIter=1000):
        self.Utility_Matrix, (self.proc1, self.proc2) = self.preprocessing_utility_matrix(Utility_Matrix)
        self.MaxIter = MaxIter
        self.latent_size = latent_size
        self.U, self.V = self.init_UV()
        self.U_best, self.V_best = self.U.copy(), self.V.copy()
        self.RMSE_tracking  = []
        self.is_trained = False
        self.M_hat = self.anti_processing(self.U_best@self.V_best)

        ## 1. Preprocessing the utility matrix
    def preprocessing_utility_matrix(self, Matrix, method=1):
        """
        The preprocessing method(s) used to process our data

        params:
            Matrix: 2D-Array
            methode (default = 1): 1. Sub the average according to the users then to the items
                                2. Sub the average according to the items then to the users
        
        return:
            The processed utility matrix
            The couple of how we processed it (so we can add this after the optimization part)
        """

        M = Matrix.copy()
        M = M*2
        proc1,proc2 = 0,0
        if method == 1:
            proc1 = np.nanmean(M, axis=0, keepdims=True)
            proc2 = np.nanmean(M, axis=1, keepdims=True)

            M = M - proc1
            M = M - proc2
        else:
            proc1 = np.nanmean(M, axis=1, keepdims=True)
            M = M - proc1

        return M, (proc1, proc2)

        ## 2. Init U and V
    def init_UV(self, randomness=False):
        """
        Randomly init the matrices for the UV-decomposition

        params:
            - randomness: if yes, init U and V randomly, else init them with the same value np.sqrt(np.nanmean(M)/latent_size)

        return:
            - random init of the matrices U and V using a normal distribution
        """
        dimension_in, dimension_out = self.Utility_Matrix.shape

        if randomness:
            U,V = np.random.normal(0, 1, (dimension_in, self.latent_size)), np.random.normal(0, 1, (self.latent_size, dimension_out))
        else:
            # According to MMDS, an interesting way to init this would be like the following:
            U,V = np.ones((dimension_in, self.latent_size))*(np.nanmean(self.Utility_Matrix)/self.latent_size), np.ones((self.latent_size,dimension_out))*(np.nanmean(self.Utility_Matrix)/self.latent_size)
        return U,V



    ######  Computational TOOOOLS
    def matmul_wo_k(self,vector, matrix, k):
        mask = [True]*vector.shape[0]
        mask[k] = False
        return vector[mask] @ matrix[mask]

    def matmul_wo_k2(self, vector, matrix, k):
        mask = [True]*vector.shape[0]
        mask[k] = False
        return matrix[: , mask]@vector[mask]

    def sparsematrix_diff(self, vector1, vector2):
        mask = ~np.isnan(vector1)
        return (vector1[mask] - vector2[mask]), mask


    def RMSE(self, m1, m2):
        """
        Function that calculates MSE 
        param:
            - m1: Ground truth
            - m2: Predicted/Estimated values
        """
        return np.sqrt(np.nanmean((m1 - m2)**2))



    # 3. Opt (tools+opt fun)
    def UV_optimization(self, verbose=False):
        """
        The main function that minimizes the RMSE
        --> The learning function
        
        param:
            - U, V (2 2D arrays: already initialised)
            - utility_matrix: (2D array) being the reference, or "the ground truth"

        return:
            U, V, MSE_tracking
        """
        RMSE_min = self.RMSE(self.Utility_Matrix, self.U@self.V)

        for it in range(self.MaxIter):
            # we want to optimize element of row 'r' and column 's'
            if (np.random.randint(0, 2) == 0):
                r = np.random.randint(0, self.U.shape[0])
                s = np.random.randint(0, self.U.shape[1])

                # computation de l'enfer
                term1 = self.matmul_wo_k(self.U[r, :], self.V, s)
                term2, mask = self.sparsematrix_diff(self.Utility_Matrix[r,:], term1)
                top = self.V[s][mask]@term2
                bottom = self.V[s, :]@self.V[s, :]

                # update
                if np.isnan(top/bottom) == False:
                    self.U[r,s] = top/bottom

            else:
                r = np.random.randint(0, self.V.shape[0])
                s = np.random.randint(0, self.V.shape[1])

                # computation de l'enfer
                term1 = self.matmul_wo_k2(self.V[:, s], self.U, r)
                term2, mask = self.sparsematrix_diff(self.Utility_Matrix[:,s], term1)
                top = self.U[:,r][mask]@term2
                bottom = self.U[:, r]@self.U[:, r]

                # update
                if np.isnan(top/bottom) == False:
                    self.V[r,s] = top/bottom

            rmse = self.RMSE(self.Utility_Matrix, self.U@self.V)
            self.RMSE_tracking.append(rmse)
            
            if rmse < RMSE_min:
                RMSE_min = rmse
                self.U_best = self.U.copy()
                self.V_best = self.V.copy()
            
                if verbose:
                    print("\n\t ---- Itteration n°",it," --- MSE=",mse)
        self.is_trained = True
        self.RMSE_tracking = np.array(self.RMSE_tracking)
        self.M_hat = self.anti_processing(self.U_best@self.V_best)

    ## 4. Undo the preprocessing
    def anti_processing(self, matrix):
        """
        params:
            - matrix (2D array)

        returns:
            - matrix (2D array)
        """
        
        m = matrix.copy()


        self.proc2[np.isnan(self.proc2)] = 0
        m += self.proc2

        self.proc1[np.isnan(self.proc1)] = 0
        m += self.proc1

        minn = m.min()
        maxx = m.max()
                # we want to have our 
        return ((m/2 - minn)*(5))/(maxx-minn)

    def getU(self):
        return self.U_best

    def getV(self):
        return self.V_best

    def getM_hat(self):
        return self.M_hat

    def get_RMSE_tracking(self):
        return self.RMSE_tracking

    def recommend_item(self, user_id):
        return self.M_hat[user_id]



class Config_A:
    def __init__(self, Utility_Matrix, latent_size=20, MaxIter=1000, test_size=0):
        self.UV_decomposition = UV_decomposition(Utility_Matrix, latent_size, MaxIter)
        self.latent_size = latent_size
        self.MaxIter = MaxIter
        self.test_size = test_size
    def opt(self, verbose=False):
        print("A: Starting the optimization part...")
        self.UV_decomposition.UV_optimization(verbose=False)
        print("Finished !")

    def getM_hat(self):
        return self.UV_decomposition.getM_hat()

    def get_RMSE_tracking(self):
        return self.UV_decomposition.get_RMSE_tracking()

    def save_rmse_tracking(self):
        print("saving points...")
        self.get_RMSE_tracking().tofile(str(self.latent_size)+"_"+str(self.test_size)+"_"+str(self.MaxIter)+"_rmseTracking.dat")
        print("saved as" + str(self.latent_size)+"_"+str(self.test_size)+"_"+str(self.MaxIter)+"_rmseTracking.dat")
    
    def recommend_k_item(self, userId, mask=None, k=10):
      arr = self.getM_hat()[userId]  
      if mask!=None:
        arr = arr[mask]
      ind = np.flip(np.argpartition(arr, -k)[-k:])
      return arr[ind]



class Config_B:
    def __init__(self, Utility_Matrix, latent_size=[20, 20, 20], MaxIter=[1000,1000,1000]):
        self.UV_decomposition1 = UV_decomposition(Utility_Matrix, latent_size[0], MaxIter[0])
        self.UV_decomposition2 = UV_decomposition(Utility_Matrix, latent_size[1], MaxIter[1])
        self.UV_decomposition3 = UV_decomposition(Utility_Matrix, latent_size[2], MaxIter[2])

    def opt(self, verbose=False):
        print("B: Starting the optimization part...")
        print("\tUV Decomposition 1:")
        self.UV_decomposition1.UV_optimization(verbose=False)
        print("\tUV Decomposition 2:")
        self.UV_decomposition2.UV_optimization(verbose=False)
        print("\tUV Decomposition 3:")
        self.UV_decomposition3.UV_optimization(verbose=False)
        print("Finished !")

    def getM_hat(self):
        return (self.UV_decomposition1.getM_hat() + self.UV_decomposition2.getM_hat() + self.UV_decomposition3.getM_hat())/3

    def get_RMSE_tracking(self):
        return (self.UV_decomposition1.get_RMSE_tracking()+self.UV_decomposition2.get_RMSE_tracking()+self.UV_decomposition3.get_RMSE_tracking())/3

    def save_rmse_tracking(self):
        print("saving points...")
        self.get_RMSE_tracking().tofile(str(self.latent_size)+"_"+str(self.MaxIter)+"_"+str(self.RMSE_tracking[-1])+"_rmseTracking.dat")
        print("saved as" + str(self.latent_size)+"_"+str(self.MaxIter)+"_"+str(self.RMSE_tracking[-1])+"_rmseTracking.dat")

    def recommend_k_item(self, userId, mask=None, k=10):
      arr = self.getM_hat()[userId]  
      if mask is not None:
        arr = arr[mask]
      ind = np.flip(np.argpartition(arr, -k)[-k:])
      return ind