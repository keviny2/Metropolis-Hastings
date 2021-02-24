import numpy as np
import scipy.special as scipy
import matplotlib.pyplot as plt

# param = [n, alpha, beta]

def sample_Beta_Binomial(n, a, b, size=None):
    p = np.random.beta(a, b, size=size)
    r = np.random.binomial(n, p)

    return r


def beta_binom_log_likelihood(param, data):
    term1 = np.log(scipy.comb(param[0], data))
    term2 = np.log(scipy.beta(data+param[1], param[0]-data+param[2]))
    term3 = np.log(scipy.beta(param[1], param[2]))
    return np.sum(term1 + term2 - term3)

    # term2 = np.log(scipy.beta(param[0] - data + param[2], data + param[1]))  ~ this was the intentional bug


def metropolis_hastings(param_init, data, num_iter):
    curr_param = param_init
    accepted = []
    rejected = []

    for i in range(num_iter):
        print("Iteration ", i)
        new_proposal = [np.random.normal(curr_param[0], 0.5, (1,)),
                        np.random.normal(curr_param[1], 0.5, (1,)),
                        np.random.normal(curr_param[2], 0.5, (1,))]

        old_likelihood = beta_binom_log_likelihood(curr_param, data)
        new_likelihood = beta_binom_log_likelihood(new_proposal, data)

        ratio = np.exp(new_likelihood - old_likelihood)
        rand = np.random.uniform(0, 1)

        if rand < ratio:
            curr_param = new_proposal
            accepted.append(new_proposal)

        else:
            rejected.append(new_proposal)

    return accepted, rejected



if __name__ == "__main__":
    np.random.seed(123)

    # ANSWER: We can test the implementation by sampling from a Beta-Binomial distribution and
    #         run MH on the data sample. The algorithm should converge to the true parameters.
    data = sample_Beta_Binomial(75, 3, 2, 1000)
    param = [200, 1, 1]
    num_iter = 100000

    accepted, rejected = metropolis_hastings(param, data, num_iter)
    accepted = accepted[int(len(accepted)/4):]
    rejected = rejected[int(len(rejected)/4):]

    n_accept = [sublist[0] for sublist in accepted]
    alpha_accept = [sublist[1] for sublist in accepted]
    beta_accept = [sublist[2] for sublist in accepted]

    n_reject = [sublist[0] for sublist in rejected]
    alpha_reject = [sublist[1] for sublist in rejected]
    beta_reject = [sublist[2] for sublist in rejected]

    plt.hist(n_accept, bins='auto')
    plt.show()
    plt.hist(alpha_accept, bins='auto')
    plt.show()
    plt.hist(beta_accept, bins='auto')
    plt.show()







