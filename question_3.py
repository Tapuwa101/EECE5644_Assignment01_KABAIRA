from pathlib import Path
from io import BytesIO
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from sklearn.decomposition import PCA

HERE = Path(__file__).parent
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)


def load_wine():
    table = pd.read_csv(HERE / 'winequality-white.csv', sep=';')
    X = table.drop(columns='quality').to_numpy(float)
    y = table['quality'].to_numpy(int)
    return X, y


def load_har():
    # Read the four needed files directly from the original ZIP.
    with zipfile.ZipFile(HERE / 'UCI HAR Dataset.zip') as archive:
        def read_file(short_name):
            file_name = next(n for n in archive.namelist()
                             if n.endswith('/' + short_name))
            return np.loadtxt(BytesIO(archive.read(file_name)))
        X = np.vstack((read_file('train/X_train.txt'),
                       read_file('test/X_test.txt')))
        y = np.r_[read_file('train/y_train.txt'),
                  read_file('test/y_test.txt')].astype(int)
    return X, y


def classify(X, y, dataset_name):
    N, features = X.shape
    classes = np.unique(y)
    log_scores = np.zeros((N, len(classes)))
    alpha = 0.02

    for j, label in enumerate(classes):
        samples = X[y == label]
        count = len(samples)
        prior = count / N
        mean = samples.mean(axis=0)
        centered = samples - mean
        covariance = centered.T @ centered / count

        # Add a small amount to the covariance diagonal for stability.
        eigenvalues = np.linalg.eigvalsh(covariance)
        tolerance = eigenvalues.max() * features * np.finfo(float).eps
        # For wine, discard extremely small eigenvalues as numerical noise.
        # HAR uses a scale-dependent numerical tolerance.
        rank = np.sum(eigenvalues > 1e-6) if dataset_name == 'wine' else np.sum(eigenvalues > tolerance)
        ridge = max(alpha * np.trace(covariance) / max(rank, 1), 1e-8)
        regularized = covariance + ridge * np.eye(features)

        # Gaussian log density plus log class prior.
        chol, lower = cho_factor(regularized, lower=True, check_finite=False)
        differences = X - mean
        solved = cho_solve((chol, lower), differences.T, check_finite=False).T
        distances = np.sum(differences * solved, axis=1)
        log_det = 2 * np.log(np.diag(chol)).sum()
        log_scores[:, j] = np.log(prior) - 0.5 * (log_det + distances)
        print(f'{dataset_name}: class {label}, count={count}, '
              f'prior={prior:.5f}, ridge={ridge:.8g}')

    predictions = classes[np.argmax(log_scores, axis=1)]
    # Rows are predicted classes; columns are actual classes.
    counts = np.array([[( (predictions == i) & (y == j)).sum()
                        for j in classes] for i in classes])
    errors = np.sum(predictions != y)
    print(f'\n{dataset_name}: {errors} errors / {N} samples = {errors/N:.6f}')
    print('Labels:', classes)
    print('Counts (rows=decision, columns=true):\n', counts)
    print('P(D=i | L=j):\n', np.round(counts/counts.sum(axis=0), 5))
    np.savetxt(OUT / f'{dataset_name}_counts.csv', counts, delimiter=',', fmt='%d')

    # PCA only for plotting: all original features are used for classification.
    pca = PCA(n_components=2, svd_solver='randomized', random_state=5644)
    points = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    for label in classes:
        use = y == label
        plt.scatter(points[use, 0], points[use, 1], s=4,
                    alpha=0.4, label=str(label))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'{dataset_name}: PCA projection')
    plt.legend(title='True label', markerscale=3)
    plt.tight_layout()
    plt.savefig(OUT / f'{dataset_name}_pca.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    classify(*load_wine(), 'wine')
    classify(*load_har(), 'har')
