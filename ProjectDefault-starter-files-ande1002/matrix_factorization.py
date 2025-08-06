import matplotlib
import skopt
from scipy.interpolate import make_interp_spline
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import numpy as np
import item_response
from skopt.space import Real, Integer
from skopt.plots import plot_convergence, plot_evaluations
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


def als_with_modification(train_data, k, lr, num_iteration, theta, beta):
    """Performs ALS algorithm, here we use the iterative solution - SGD
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :param theta: A zeroed matrix representing student skill level
    :param beta: A zeroed matrix representing question difficulty
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
    bias = np.mean(train_data["is_correct"])

    for it in range(num_iteration):
        i = np.random.choice(len(train_data["user_id"]))
        user_id = train_data["user_id"][i]
        question_id = train_data["question_id"][i]
        score = train_data["is_correct"][i]

        pred = bias + theta[user_id] + beta[question_id] + np.dot(u[user_id], z[question_id])
        error = score - pred

        u_old = u[user_id].copy()

        u[user_id] += lr * error * z[question_id]
        z[question_id] += lr * error * u_old

        theta[user_id] += lr * error
        beta[question_id] += lr * error

    final_predictions = bias + theta[:, np.newaxis] + beta[np.newaxis, :] + np.dot(u, z.T)
    return final_predictions

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


def helper_evaluate(k, lr, num_iter, train_data, val_data, theta, beta):
    """
    :param train_data: The training data used to evaluate training accuracy based on
    hyper_parameters
    :param k: int
    :param lr: float
    :param num_iter: int
    :return: A float representing negative accuracy (gp minimize minimizes accuracy)
    """

    accuracy = sparse_matrix_evaluate(val_data, als_with_modification(train_data, k, lr, num_iter,
                                                                      theta, beta))
    return -accuracy


def find_optimal_hyperparameters(search_space, iterations, train_data, val_data, theta, beta):
    """
     Designed to predict optimal hyperparameters using Bayesian Optimization.

    :param search_space: A list of tuples, each specifying the range of values to be considered
    for a specific hyperparameter, e.g. [Integer(10, 100, name='k'), Real(0.1, 1.0, name='lr')]
    :param iterations: Number of iterations to preform
    :param train_data: The training data
    returns:
        res : OptimizeResult, a scipy object
            The optimization result returned as an OptimizeResult object.
            Important attributes are:
            x : list
                The location of the minimum (i.e., the best hyperparameter values found).
            fun : float
                The function value at the minimum (i.e., the best score).
            x_iters : list of lists
                The location of function evaluation for each iteration (the hyperparameter values tried).
            func_vals : array
                The function value for each iteration (the scores achieved for each trial).
            space : Space
                The optimization space.
            specs : dict
                The call specifications.
            rng : RandomState instance
                State of the random state at the end of minimization.

    Note: This functions depends on helper_evaluate which acts as a black box to evaluate the
    accuracy function based on a set of hyperparameters
    """

    function_to_minimize = lambda hyperparameters: helper_evaluate(
        hyperparameters[0],
        hyperparameters[1],
        hyperparameters[2],
        train_data,
        val_data,
        theta,
        beta

    )

    return skopt.gp_minimize(function_to_minimize, search_space, n_calls=iterations)


def main():
    train_matrix = load_train_sparse("./data").toarray()
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    k_values = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    learning_rates = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    iterations = [500, 1000, 5000, 8000, 10000, 15000]
    als_with_modification_accuracy = []
    als_accuracy = []
    hyperparameters = []
    theta = np.zeros(train_matrix.shape[0])
    beta = np.zeros(train_matrix.shape[1])
    for k in k_values:
        for lr in learning_rates:
            for num_iter in iterations:
                mat = als_with_modification(train_data, k, lr, num_iter, theta, beta)
                acc = sparse_matrix_evaluate(val_data, mat)
                als_with_modification_accuracy.append(acc)
                mat = als(train_data, k, lr, num_iter)
                acc = sparse_matrix_evaluate(val_data, mat)
                als_accuracy.append(acc)
                hyperparameters.append((k, lr, num_iter))

    plt.figure(figsize=(8, 6))
    plt.scatter(als_accuracy, als_with_modification_accuracy, alpha=0.7)
    plt.plot([0, 1], [0, 1], 'r--', label='y = x (equal performance)')

    plt.xlabel('Original ALS Accuracy')
    plt.ylabel('Modified ALS Accuracy')
    plt.title('ALS vs ALS with Modification')
    plt.legend()
    plt.grid(True)
    plt.show()

    x = range(len(als_accuracy))  # indices for each run

    plt.figure(figsize=(12, 6))
    plt.plot(x, als_accuracy, label='ALS Original', marker='o')
    plt.plot(x, als_with_modification_accuracy, label='ALS Modified', marker='o')

    plt.xlabel('Run Index (Hyperparameter Combination)')
    plt.ylabel('Validation Accuracy')
    plt.title('ALS vs Modified ALS Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    x = np.arange(len(als_accuracy))

    x_smooth = np.linspace(x.min(), x.max(), 500)

    als_smooth = make_interp_spline(x, als_accuracy)(x_smooth)
    als_mod_smooth = make_interp_spline(x, als_with_modification_accuracy)(x_smooth)

    plt.figure(figsize=(14, 6))
    plt.plot(x_smooth, als_smooth, label='ALS Original', linewidth=2)
    plt.plot(x_smooth, als_mod_smooth, label='ALS Modified', linewidth=2)

    plt.xlabel('Hyperparameter Combination Index')
    plt.ylabel('Validation Accuracy')
    plt.title('ALS vs ALS with Modification Accuracy (Smooth Curve)')
    plt.legend()
    plt.grid(True)
    plt.show()


    # #####################################################################
    # #                                                          #
    # # (SVD) Try out at least 5 different k and select the best k        #
    # # using the validation set.                                         #
    # #####################################################################
    # arr = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
    # best_k = 0
    # highest_accuracy = 0.0
    # for k in arr:
    #
    #     matrix = svd_reconstruct(train_matrix, k)
    #     accuracy = sparse_matrix_evaluate(val_data, matrix)
    #
    #     print(f"k = {k}: Validation Accuracy = {accuracy:.4f}")
    #
    #     if accuracy > highest_accuracy:
    #         highest_accuracy = accuracy
    #         best_k = k
    #
    # print(f"Best k selected based on validation accuracy is: {best_k}")
    # print(f"Highest validation accuracy: {highest_accuracy}")
    #
    # final_reconstructed_matrix = svd_reconstruct(train_matrix, best_k)
    #
    # final_test_accuracy = sparse_matrix_evaluate(test_data, final_reconstructed_matrix)
    #
    # print(f"Final Test Accuracy with k={best_k}: {final_test_accuracy:.4f}")
    #
    # #####################################################################
    # #                       END OF YOUR CODE                            #
    # #####################################################################
    #
    # #####################################################################
    #
    # # (ALS) Try out at least 5 different k and select the best k        #
    # # using the validation set.                                         #
    # #####################################################################
    # k_values = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    # learning_rates = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    # iterations = [500, 1000, 5000, 8000, 10000, 15000]
    #
    # best_k = None
    # best_lr = None
    # best_iter = None
    # best_val_acc = 0
    # best_train_acc = 0
    #
    # for k in k_values:
    #     for lr in learning_rates:
    #         for num_iter in iterations:
    #             mat = als(train_data, k, lr, num_iter)
    #             val_acc = sparse_matrix_evaluate(val_data, mat)
    #             train_acc = sparse_matrix_evaluate(train_data, mat)
    #
    #             if val_acc > best_val_acc:
    #                 best_val_acc = val_acc
    #                 best_k = k
    #                 best_lr = lr
    #                 best_iter = num_iter
    #             if train_acc > best_train_acc:
    #                 best_train_acc = train_acc
    #
    # print("Learning the representations U and Z using ALS with SGD...")
    # print("---------------------------------------------------------")
    # print(f"Best k: {best_k}")
    # print(f"Best learning rate: {best_lr}")
    # print(f"Best iterations: {best_iter}")
    # print(f"Best validation accuracy: {best_val_acc}")
    # print(f"Best training accuracy: {best_train_acc}")
    # mat_best = (als(train_data, best_k, best_lr, best_iter))
    # acc_test = sparse_matrix_evaluate(test_data, mat_best)
    # print(f"Best test accuracy: {acc_test}")
    #
    # mat, train_losses, val_losses = als_with_tracking(train_data, val_data, best_k, best_lr,
    #                                                   best_iter)
    # n_train = len(train_data["is_correct"])
    # n_val = len(val_data["is_correct"])
    #
    # train_losses = [loss / n_train for loss in train_losses]
    # val_losses = [loss / n_val for loss in val_losses]
    #
    # track_every = 100
    # num_points = len(train_losses)
    # iterations = np.arange(1, num_points + 1) * track_every
    #
    # plt.figure(figsize=(8, 5))
    # plt.plot(iterations, train_losses, label="Training Loss")
    # plt.plot(iterations, val_losses, label="Validation Loss")
    # plt.xlabel("Iterations")
    # plt.ylabel("Squared Error Loss")
    # plt.title("Training and Validation Loss vs Iterations")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # #####################################################################
    # # #                       END OF YOUR CODE                            #
    # # #####################################################################
    # #
    # # #####################################################################
    # # # In this part we try to optimize hyperparameters using our         #
    # # # find_optimal_hyperparameters function we will try 1000 iterations #
    # # #####################################################################
    # # First create our search space
    # search_space = [
    #     Integer(1, 1000, name='k'),
    #     Real(0.0001, 0.3, name='lr'),  # Keep at less than 0.3 or becomes unstable
    #     Integer(1, 1000, name='num_iter'),
    # ]
    # # Now call our function
    # theta = np.zeros(train_matrix.shape[0])
    # beta = np.zeros(train_matrix.shape[1])
    # result = find_optimal_hyperparameters(search_space, 100, train_data, val_data, theta, beta)
    #
    # print(f"Optimal hyperparameters: k={result.x[0]}, lr={result.x[1]}, num_iter={result.x[2]}")
    # print(f"Best score achieved: {-result.fun}")
    #
    # # Plot the convergence
    # _ = plot_convergence(result)
    # plt.show()
    #
    # fig = plt.figure(figsize=(10, 8))
    #
    # _ = plot_evaluations(result, ax=fig.gca())
    #
    # for ax in fig.get_axes():
    #     for label in ax.get_yticklabels():
    #         label.set_rotation(45)
    #
    # plt.tight_layout()
    # plt.show()

    # mat = als_with_modification(train_data, 650, 0.017, 3000000, theta, beta)
    # val_acc = sparse_matrix_evaluate(val_data, mat)
    # print(val_acc)


if __name__ == "__main__":
    main()
