__author__ = 'karthikb'

from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import matplotlib.pyplot as plt

def feature_selection_plot(x,y,no_of_features =1,min_range = None):
    model = ExtraTreesClassifier(compute_importances=True)
    model.fit(x,y)
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)[::-1]
    print "Feature importance:"
    print sorted_idx
    i = 1
    unused_columns = []
    for f,w in zip(x.columns[sorted_idx], feature_importance[sorted_idx]):
        if w == 0:
            unused_columns.append(f)
        print "%d) %s : %d" % (i, f, w)
        i+=1
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    nb_to_display = no_of_features
    plt.barh(pos[:nb_to_display], feature_importance[sorted_idx][:nb_to_display], align='center')
    plt.yticks(pos[:nb_to_display], x.columns[sorted_idx][:nb_to_display])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    return x.drop(unused_columns,axis=1)
