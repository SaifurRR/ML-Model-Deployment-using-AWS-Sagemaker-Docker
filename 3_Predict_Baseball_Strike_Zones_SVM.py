import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import product
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from players import jose_altuve, david_ortiz  

#meshgrid
def make_meshgrid(ax, h=.02):
    x_min, x_max = float(ax.get_xlim()[0]), float(ax.get_xlim()[1])
    y_min, y_max = float(ax.get_ylim()[0]), float(ax.get_ylim()[1])
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

#plot contours    
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return ax.contourf(xx, yy, Z, **params)

#draw boundary    
def draw_boundary(ax, clf):
    xx, yy = make_meshgrid(ax)
    return plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

# dataset preprocessing
def preprocess_dataset(dataset):
    dataset['type'] = dataset['type'].map({'S': 1, 'B': 0})
    dataset = dataset.dropna(subset=['plate_x', 'plate_z', 'type'])
    return dataset

# train SVM and compute score
def train_svm(dataset, gamma, C):
    training_set, validation_set = train_test_split(dataset, random_state=1)
    scores = []
    best_classifier = None
    best_score = -np.inf
    for g, c in product(gamma, C):
        classifier = SVC(kernel='rbf', gamma=g, C=c)
        classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
        score = classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type'])
        scores.append(score)
        if score > best_score:
            best_score = score
            best_classifier = classifier
    return best_classifier, scores

# plot pitch locations and SVM boundary
def plot_pitch(dataset, classifier):
    fig, ax = plt.subplots()
    ax.scatter(dataset.plate_x, dataset.plate_z, c=dataset.type, cmap=plt.cm.coolwarm, alpha=0.25)
    draw_boundary(ax, classifier)
    ax.set_xlabel('plate_x')
    ax.set_ylabel('plate_z')
    ax.set_title('Pitch Location')
    plt.show()

# optimize gamma and C parameters
def optimize_parameters(dataset, gamma, C):
    gamma = list(gamma)
    C = list(C)    
    _, scores = train_svm(dataset, gamma, C)
    max_score = max(scores)
    best_index = scores.index(max_score)
    best_gamma, best_C = gamma[best_index // len(C)], C[best_index % len(C)]
    print(f"Best SVM Score: {max_score:.4f} at gamma={best_gamma}, C={best_C}")
    # Reshape scores into a 2D array for heatmap
    scores_matrix = np.array(scores).reshape(len(gamma), len(C))
    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(scores_matrix, xticklabels=C, yticklabels=gamma, cmap="coolwarm", annot=True)
    plt.xlabel("C Values")
    plt.ylabel("Gamma Values")
    plt.title("SVM Score Heatmap")
    plt.show()

if __name__ == "__main__":
    gamma_range = np.arange(0.1, 2.0, 0.1)
    C_range = np.arange(1, 10, 1)
    dataset = preprocess_dataset(jose_altuve)
    classifier, _ = train_svm(dataset, gamma_range, C_range)
    plot_pitch(dataset, classifier)
    optimize_parameters(dataset, gamma_range, C_range)
