import matplotlib
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('TkAgg')

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def svd_reconstruct(matrix, k):
    """Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i] - np.sum(u[data["user_id"][i]] * z[q])) ** 2.0
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    #                                            #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = np.random.choice(len(train_data["question_id"]), 1)[0]
    # So now we need to get the random entry

    outcome = train_data["is_correct"][i]
    user_id = train_data["user_id"][i]
    question_id = train_data["question_id"][i]

    s = user_id
    k = question_id
    error = outcome - np.dot(u[s], z[k].T)

    u_old = u[s].copy()
    u[s] += lr * error * z[k]
    z[k] += lr * error * u_old
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration):
    """Performs ALS algorithm, here we use the iterative solution - SGD
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(
        low=0, high=1 / np.sqrt(k), size=(len(set(train_data["user_id"])), k)
    )
    z = np.random.uniform(
        low=0, high=1 / np.sqrt(k), size=(len(set(train_data["question_id"])), k)
    )

    #####################################################################
    #                                                      #
    # Implement the function as described in the docstring.             #
    #####################################################################

    for it in range(num_iteration):
        i = np.random.choice(len(train_data["user_id"]))
        user_id = train_data["user_id"][i]
        question_id = train_data["question_id"][i]
        score = train_data["is_correct"][i]

        pred = np.dot(u[user_id], z[question_id])
        error = score - pred

        u_old = u[user_id].copy()

        u[user_id] += lr * error * z[question_id]
        z[question_id] += lr * error * u_old

    mat = np.dot(u, z.T)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat


def als_with_tracking(train_data, val_data, k, lr, num_iterations):
    num_users = len(set(train_data["user_id"]))
    num_questions = len(set(train_data["question_id"]))

    u = np.random.uniform(0, 1/np.sqrt(k), (num_users, k))
    z = np.random.uniform(0, 1/np.sqrt(k), (num_questions, k))

    train_losses = []
    val_losses = []

    for it in range(num_iterations):
        idx = np.random.choice(len(train_data["user_id"]))
        user = train_data["user_id"][idx]
        question = train_data["question_id"][idx]
        score = train_data["is_correct"][idx]

        pred = np.dot(u[user], z[question])
        error = score - pred

        u_old = u[user].copy()
        u[user] += lr * error * z[question]
        z[question] += lr * error * u_old

        if it % 100 == 0:
            train_losses.append(squared_error_loss(train_data, u, z))
            val_losses.append(squared_error_loss(val_data, u, z))

    mat = np.dot(u, z.T)
    return mat, train_losses, val_losses


def main():
    train_matrix = load_train_sparse("./data").toarray()
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    #                                                          #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    arr = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
    best_k = 0
    highest_accuracy = 0.0
    for k in arr:

        matrix = svd_reconstruct(train_matrix, k)
        accuracy = sparse_matrix_evaluate(val_data, matrix)

        print(f"k = {k}: Validation Accuracy = {accuracy:.4f}")

        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_k = k

    print(f"Best k selected based on validation accuracy is: {best_k}")
    print(f"Highest validation accuracy: {highest_accuracy}")

    final_reconstructed_matrix = svd_reconstruct(train_matrix, best_k)

    final_test_accuracy = sparse_matrix_evaluate(test_data, final_reconstructed_matrix)

    print(f"Final Test Accuracy with k={best_k}: {final_test_accuracy:.4f}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################

    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    k_values = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    learning_rates = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    iterations = [500, 1000, 5000, 8000, 10000, 15000]

    best_k = None
    best_lr = None
    best_iter = None
    best_val_acc = 0
    best_train_acc = 0

    for k in k_values:
        for lr in learning_rates:
            for num_iter in iterations:
                mat = als(train_data, k, lr, num_iter)
                acc = sparse_matrix_evaluate(val_data, mat)
                train_acc = sparse_matrix_evaluate(train_data, mat)

                if acc > best_val_acc:
                    best_val_acc = acc
                    best_k = k
                    best_lr = lr
                    best_iter = num_iter
                if train_acc > best_train_acc:
                    best_train_acc = train_acc

    print("Learning the representations U and Z using ALS with SGD...")
    print("---------------------------------------------------------")
    print(f"Best k: {best_k}")
    print(f"Best validation accuracy: {best_val_acc}")
    print(f"Best training accuracy: {best_train_acc}")
    print(f"Best learning rate: {best_lr}")
    print(f"Best iterations: {best_iter}")
    mat_best = (als(train_data,best_k, best_lr, best_iter))
    acc_test = sparse_matrix_evaluate(test_data, mat_best)
    print(f"Best test accuracy: {acc_test}")

    mat, train_losses, val_losses = als_with_tracking(train_data, val_data, best_k, best_lr,
                                                      best_iter)
    n_train = len(train_data["is_correct"])
    n_val = len(val_data["is_correct"])

    train_losses = [loss / n_train for loss in train_losses]
    val_losses = [loss / n_val for loss in val_losses]

    track_every = 100
    num_points = len(train_losses)
    iterations = np.arange(1, num_points + 1) * track_every

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, train_losses, label="Training Loss")
    plt.plot(iterations, val_losses, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Squared Error Loss")
    plt.title("Training and Validation Loss vs Iterations")
    plt.legend()
    plt.grid(True)
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
