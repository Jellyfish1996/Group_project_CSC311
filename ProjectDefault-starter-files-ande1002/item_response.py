from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.0
    for i in range(len(data["user_id"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        c = data["is_correct"][i]

        z = theta[u] - beta[q]

        if c == 1:
            log_lklihood -= np.logaddexp(0, -z)
        else:
            log_lklihood -= np.logaddexp(0, z)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    d_theta = np.zeros_like(theta)
    d_beta = np.zeros_like(beta)

    for i in range(len(data["user_id"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        c = data["is_correct"][i]

        z = theta[u] - beta[q]
        p = sigmoid(z)

        d_theta[u] += c - p
        d_beta[q] += p - c

    theta += lr * d_theta
    beta += lr * d_beta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    num_users = max(data["user_id"]) + 1  # assuming zero-based IDs
    num_questions = max(data["question_id"]) + 1
    theta = np.random.normal(0, 1, size=num_users)
    beta = np.random.normal(0, 1, size=num_questions)

    val_acc_lst = []
    train_lld_lst = []
    val_lld_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_lld_lst.append(neg_lld)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        val_lld = neg_log_likelihood(val_data, theta, beta)
        val_lld_lst.append(val_lld)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_lld_lst, val_lld_lst


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    train_lls = []
    val_lls = []
    val_accs = []

    lr = 0.001
    # iterations = np.arange(0, 900)

    # theta, beta, val_acc_lst,train_lld_lst , val_lld_lst = irt(train_data, val_data,lr,900)

    # plt.plot(iterations, train_lld_lst, label="Training Log-Likelihood")
    # plt.plot(iterations,val_lld_lst, label = "Validation Log-Likelihood")
    # plt.xlabel("Iteration")
    # plt.ylabel("Log-Likelihood")
    # plt.title("Training Curve for Number of Iterations")
    # plt.legend()
    # plt.show()

    # plt.plot(iterations, val_acc_lst, label="Validation Accuracy")
    # plt.xlabel("Iteration")
    # plt.ylabel("Validation Score")
    # plt.title("Validation Scores for Number of Iterations")
    # plt.legend()
    # plt.show()

    # final hyperparameters and test
    theta, beta, val_acc_lst, train_lld_lst, val_lld_lst = irt(train_data, val_data, lr, 500)
    final_val_score = val_acc_lst[-1]
    test_score = evaluate(test_data, theta, beta)
    print("Test score: ", test_score)
    print("Final Validation score:  ", final_val_score)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)
    #
    #####################################################################

    thetas = []
    probs = []
    theta_range = np.linspace(-3, 3, 100)

    for i in range(3):
        u = test_data["user_id"][i]
        q = test_data["question_id"][i]
        c = test_data["is_correct"][i]

        probs = [sigmoid(t - beta[q]) for t in theta_range]
        plt.plot(theta_range, probs, label=f"Question {q}")

    plt.xlabel("Theta (student ability)")
    plt.ylabel("Probability of Correct Response")
    plt.title("IRT Item Characteristic Curves")
    plt.legend()
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
