#%%

import glob
import os.path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import SplineTransformer
from sklearn.model_selection import train_test_split
from cloudio import load_all_sessions

#%%

def model_cv(X, y, random_state=42, test_size=0.2):
    """
    fit logistic regression model with cross-validation but no regularization
    return fitted model and test log-likelihood/accuracy
    """
    clf = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=random_state, C=np.inf)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    # get log-likelihood of test set
    pred_probs = clf.predict_proba(X_test)
    log_likelihood = -log_loss(y_test, pred_probs, normalize=True)
    return { 'log_likelihood': log_likelihood, 'accuracy': accuracy, 'model': clf }

def logreg_model_spline(df, random_state=None):
    """
    fit logistic regression model with spline interaction terms to df
    """
    X1 = df['dist_L1'].values
    X2 = df['dist_R1'].values
    X3 = df['dist_L2'].values
    X4 = df['dist_R2'].values
    X5 = df['heightOnScreen'].values
    d1 = X1 - X2 # one-step distance difference
    d2 = X1 + X3 - X2 - X4 # two-step distance difference
    z = X5
    
    # get spline basis functions for z
    spline = SplineTransformer(n_knots=3, degree=2, include_bias=False)
    Bz = spline.fit_transform(z.reshape(-1, 1))   # (n, k) basis functions of z

    # project d1 and d2 onto each basis
    feat_x1 = Bz * d1[:, None]
    feat_x2 = Bz * d2[:, None]

    # build design matrix
    X = np.hstack([d1[:, None], d2[:, None], feat_x1, feat_x2])
    y = df['choice'].values
    
    return model_cv(X, y, random_state=random_state)

def fit_models_per_subj(df, random_state=None, normalize=False):
    """
    fit logistic regression models to df for each subject and compare their performance
    """
    results = {}
    for subj in df['subject_id'].unique():
        df_subj = df[df['subject_id'] == subj]
        results[subj] = {}
        results[subj]['+spline'] = logreg_model_spline(df_subj, random_state=random_state)
    return results

def plot_spline_weights(df, res, n_knots=3, mdl_name='+spline'):
    ncols = np.ceil(np.sqrt(len(res))).astype(int)
    nrows = np.ceil(len(res) / ncols).astype(int)
    ncols = 2; nrows = 1
    fig = plt.figure(figsize=(3 * ncols, 3 * nrows))
    for c, (subj, res_subj) in enumerate(res.items()):
        # visualize weights of the model
        clf = res_subj[mdl_name]['model']
        coef = clf.coef_[0]
        spline = SplineTransformer(n_knots=n_knots, degree=2, include_bias=False)

        # get height range for this subject
        dfc = df[df['subject_id'] == subj]
        z_grid = np.linspace(dfc['heightOnScreen'].min(), dfc['heightOnScreen'].max(), 200)
        Bz_grid = spline.fit_transform(z_grid.reshape(-1, 1))

        beta_d1_of_z = coef[0] + Bz_grid @ coef[2:(2 + n_knots)]   # d1 main effect + interaction terms
        beta_d2_of_z = coef[1] + Bz_grid @ coef[(2 + n_knots):(2 + 2 * n_knots)]  # d2 main effect + interaction terms

        plt.subplot(nrows, ncols, 1)
        plt.plot(z_grid, beta_d1_of_z, label='greedy')
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel('height on screen (0 is top, 1 is bottom)')
        plt.ylabel('coefficient weight')
        plt.gca().set_title('greedy weights')
        plt.subplot(nrows, ncols, 2)
        plt.plot(z_grid, beta_d2_of_z, label='planning')
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel('height on screen')
        plt.ylabel('coefficient weight')
        plt.gca().set_title('planning weights')
    plt.tight_layout()
    plt.show()

#%%

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # allow multiple data directories
    parser.add_argument('datadir', type=str, nargs='+', default=['~/Downloads/First Ten Participants', '~/Downloads/Second 5 Participants'])
    args = parser.parse_args()

    # load all json files from the specified directories
    fnms = []
    for datadir in args.datadir:
        datadir = os.path.expanduser(datadir)
        fnms.extend(glob.glob(os.path.join(datadir, '*.json')))
    df = load_all_sessions(fnms)

    res = fit_models_per_subj(df, random_state=555, normalize=False)
    plot_spline_weights(df, res)
