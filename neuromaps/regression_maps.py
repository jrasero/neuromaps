import numpy as np
from numpy import column_stack as cstack
from collections import namedtuple
from joblib import Parallel, delayed

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import r2_score
from nilearn.image import new_img_like


class BasePredictiveMaps():
        
    def fit(self, X, y):
        return
     
    def fit_predict(self, X, y):
        return
    
    def predict(self, X):
        return
    
    def _check_inputs(self, X, y):
        
        from nibabel.nifti1 import Nifti1Image
        from nilearn.image import load_img, concat_imgs
        
        if type(X)==list:
            # First, load images
            X = [load_img(img) for img in X]
            
            # check that all elements of the list have dimension 3 (should be 3D images)
            dims_X = [img.ndim!=3 for img in X]
            if np.any(dims_X):
                raise print("List of images provided, so they all should be 3D")
                
            X = concat_imgs(X)
        
        elif type(X) == Nifti1Image:
            if X.ndim!=4:
                raise print("One Nifti image, so it should be 4D")
            
            n_obs = X.shape[3]
            
            if n_obs < 3:
                print("check the results, with observations < 3 the correlations"
                     " will NaN or always 1")
        
        y = np.asarray(y)
        
        if y.ndim > 1:
            raise print("Dependent variable y should be unidimensional")
        
        return X, y

    
class RegressionMaps(BasePredictiveMaps):

    def __init__(self, n_splits=5, stratify=True, n_jobs=1, random_state=None, verbose=0):
        self.n_splits = n_splits
        self.stratify = stratify
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        
        #from tqdm import tqdm

        X, y = self._check_inputs(X, y)
        
        X_data = X.get_fdata()
        
        #TODO: Add option to include a custom mask image
        mask = np.ones(X.shape[:3], dtype=bool)
        X_masked = X_data[mask].T # Time Points x Voxels
        n_voxels = X_masked.shape[1]
        
        # Ensure NaN are zero in the masked data
        X_masked = np.nan_to_num(X_masked)
        
        
        if self.stratify:
            # Convert target to bins, so that it can be stratified within the cross-validation
            y_bin = np.digitize(y, np.quantile(y, q=np.arange(0,1, 0.2)))
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        else:
            # No stratify, so y_bin is y
            y_bin = y
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        linReg = LinearRegression()
        
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)

        results = parallel(delayed(_inverse_reg_cv)(clone(linReg), 
                                                    X_masked, y, train_index, test_index) 
                           for (train_index, test_index) in cv.split(X_masked, y_bin))
        
        y_pred_voxels, y_true = zip(*results) # Extract multiple returns from the parallel function
        y_pred_voxels = np.row_stack(y_pred_voxels)
        y_true = np.concatenate(y_true)
        
        # Compute correlation and R2 maps
        corr_voxels = np.zeros(n_voxels)
        r2_voxels = np.zeros(n_voxels)
        
        # Do not use voxels with no predictions at all
        mask_zeros = ~np.all(y_pred_voxels==0, axis=0)
        
        def func_r2(a, y_true):
            return r2_score(y_true, a)
        def func_cor(a, y_true):
            return np.corrcoef(a,y_true)[0,1]

        r2_voxels[mask_zeros] = np.apply_along_axis(func_r2, 0, y_pred_voxels[:,mask_zeros], y_true)
        # Set negative r2 values to zero, as they are not predictive
        r2_voxels[r2_voxels<0]=0
        
        corr_voxels[mask_zeros] = np.apply_along_axis(func_cor, 0, y_pred_voxels[:,mask_zeros], y_true)
        corr_voxels[corr_voxels<0]=0 # Should I do this as well?
        
        # return both predictive measures as images
        r2_image_data = np.zeros(X.shape[:3])
        r2_image_data[mask] = r2_voxels    
        self.r2_image_ = new_img_like(X, r2_image_data)

        corss_image_data = np.zeros(X.shape[:3])
        corss_image_data[mask] = corr_voxels    
        self.corrs_image_ = new_img_like(X, corss_image_data)
        
        return self

    def fit_predict(self, X, y):
                
        self.fit(X, y)
        reg_maps_results = namedtuple("RegressionMaps", ["R2_maps", "Corr_maps"])
        
        return reg_maps_results(R2_maps=self.r2_image_, Corr_maps=self.corrs_image_)
    
    def predict(self, X):
        
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self)

        reg_maps_results = namedtuple("RegressionMaps", ["R2_maps", "Corr_maps"])
        
        return reg_maps_results(R2_maps=self.r2_image_, Corr_maps=self.corrs_image_)
    
def _inverse_reg_cv(estimator, X, y, train_index, test_index):
    """
    Function to predict in a fold, fitting first
    """
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    estimator.fit(y_train[:, None], X_train)

    beta_y_on_x = np.squeeze(estimator.coef_.T)
    mu_x = np.mean(X_train, axis=0)
    mu_y = np.mean(y_train)
    var_x = np.var(X_train, axis=0)
    var_y = np.var(y_train)

    mask_nan = (var_x!=0.)

    beta = (var_y*beta_y_on_x[mask_nan])/var_x[mask_nan]
    alpha = mu_y - (mu_x[mask_nan]*beta_y_on_x[mask_nan]/var_x[mask_nan])*var_y

    y_pred_fold = np.zeros(X_test.shape)
    
    def func_pred(a, alpha, beta):
        pred = alpha + a*beta
        return pred

    y_pred_fold[:, mask_nan] = cstack([func_pred(x, a, b) 
                                       for (x,a,b) in zip(X_test[:, mask_nan].T, alpha, beta)])

    return (y_pred_fold, y_test)
