"""
P(Theta | X) = Sum_{qj} P(X | Qj=qj) * P(Qj | Theta) * P(Theta) / P(X)
"""
import os
import pickle
import numpy as np
import numpy.random as rnd
import jax
import jax.numpy as jnp
import optax
import scipy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal, entropy
from datetime import datetime
# from bcipy.signal.model.pca_rda_kde.modified_rda import ModifiedRdaModel
# from sklearn.model_selection import train_test_split
class Simulation:
    """
    Simulate the BCI experiment.
    """

    def __init__(self, target_phrase, alphabet, inquiry_length, prob_alphabet,
                 seed_inquiry, threshold):
        """
        :param alphabet list: The list of characters.
        :param inquiry_length int: The number of characters shown in
        the inquiry.
        :param prob_alphabet list: Prior probability for each character.
        :param seed int: Seed used for random number generator.
        :param generative_fn function: Generative model.
        """
        self.target_phrase = target_phrase
        self.alphabet = alphabet
        self.n_alphabet = len(alphabet)
        self.erp = 0.3 # seconds, P300 response
        self.flash_time = 0.3 # seconds
        self.sampling_rate = 50  # Hz
        self.inquiry_length = inquiry_length
        self.n_classes = inquiry_length + 1
        self.n_electrodes = 1  # Number of EEG electrodes
        self.inquiry_duration = self.flash_time * self.inquiry_length + 0.2 # seconds
        self.dim = int(self.inquiry_duration * self.sampling_rate)
        self.time_samples = np.linspace(0, self.inquiry_duration, self.dim, endpoint=False)
        self.nll = []
        self.maxiter = 50
        self.stepsize = 1e-2
        self.prob_alphabet = prob_alphabet
        self.prob_alphabet_prior = prob_alphabet.copy()
        self.seed_inquiry = seed_inquiry  
        self.seed_mean_est = 2025  
        self.seed_mean_true = 3000
        self.mult_mean = 25
        self.rng_inquiry = rnd.default_rng(seed_inquiry)
        self.vec_a = np.ones((self.n_classes, int(self.dim * (self.dim - 1) / 2))) * 1e-3
        self.vec_d = np.ones((self.n_classes, self.dim)) * 1e-3
        self.cov_mean = 5e-3
        self.l_means_true = self.generate_initial_means()
        # self.l_covariances_true = self.generate_cov_matrix(self.vec_a, self.vec_d)
        # self.l_covariances_true[0] = np.eye(self.dim) / (self.cov_mean * 2 * np.pi * 100)
        self.l_covariances_true = np.array([np.eye(self.dim) / (self.cov_mean * 2 * np.pi * 400) for i_class in range(self.n_classes)])
        self.l_means_estimate = np.zeros_like(self.l_means_true)     # initialize, update with calibration
        self.l_covariances_estimate = np.zeros_like(self.l_covariances_true)
        self.parameter_prior = multivariate_normal.pdf(np.zeros(self.dim), np.zeros(self.dim), np.eye(self.dim))
        self.threshold = threshold
        self.entropies = [[] for character in target_phrase]
        self.inquiries = [[] for character in target_phrase]
        self.measurements = [[] for character in target_phrase]
        self.posteriors = [[] for character in target_phrase]
        self.verbose = 1 
        self.written_phrase = []
        self.hamming = 0


    def plot_means(self, means):
        for i_class in range(self.n_classes):
            plt.plot(self.time_samples, means[i_class], label=f"Class {i_class}")
            plt.legend()
        plt.show()
        
    def generate_precision_matrix(self, vec_a, vec_d):
        """
        Generate precision matrix using the SVD property.
        pres = U @ V @ U.T, U = e^B, V = e^D, D = diag(lambda), B = skew_symmetric_matrix(A), A = random matrix
        """
        A = self.generate_skew_symmetric_matrix(vec_a)
        U = scipy.linalg.expm(A)
        D = np.diag(vec_d)
        V = scipy.linalg.expm(D)
        pre = U @ V @ U.T
        return pre
    
    def generate_skew_symmetric_matrix(self, vec_a):
        # len(vec_a) = dim * (dim - 1) / 2
        A = np.zeros((self.dim, self.dim))
        i_vec = 0
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                A[i, j] = vec_a[i_vec]
                A[j, i] = -vec_a[i_vec]
                i_vec += 1
        return A
    
    def generate_initial_means(self):
        """Generate initial means."""
        l_means = np.zeros((self.n_classes, self.dim))
        for i_class in range(self.n_classes):
            if not i_class:
                continue
            l_means[i_class] = multivariate_normal.pdf(
                self.time_samples,
                mean=i_class * self.erp,
                cov=self.cov_mean
            )

        return l_means

    def generate_cov_matrix(self, vec_a, vec_d):
        l_precisions = np.zeros((self.n_classes, self.dim, self.dim))
        for i_class in range(self.n_classes):
            l_precisions[i_class] = self.generate_precision_matrix(vec_a[i_class], vec_d[i_class])
        
        return l_precisions
    
    def generate_inquiry(self):
        """
        Generate inquiry given prior probability for each character.
        
        :return inquiry list: Ordered list with the characters shown in
        inquiry.
        """
        inquiry_indices = self.rng_inquiry.choice(
            self.n_alphabet,
            size=self.inquiry_length,
            replace=False,
            p=self.prob_alphabet
            )
        inquiry = [self.alphabet[i_inquiry] for i_inquiry in inquiry_indices]

        return inquiry
    
    def generate_measurement(self, character_position):
        """
        Generate brain activity given that the target character is at the ith
        location in inquiry.

        :return measurement list: Measurement given target character and
        inquiry.
        """
        rng_measurement = rnd.default_rng()
        measurement = rng_measurement.multivariate_normal(
                self.l_means_true[character_position],
                self.l_covariances_true[character_position]
                )

        return measurement
    
    def calculate_entropy(self, probability_dist, epsilon=1e-8):
        """
        Calculate Shannon entropy of posterior probability.
        """
        # NOTE: Without deep copying the argument probabaility_dist, this method
        # makes changes to the original list.
        # To keep the original list, use probability_dist = probability_dist.copy()
        for i_prob, prob in enumerate(probability_dist):
            if prob < epsilon:
                probability_dist[i_prob] = epsilon
        probability_dist /= probability_dist.sum()
        
        return -np.dot(probability_dist, np.log2(probability_dist))
    
    def calculate_hamming_distance(self):
        """
        Calculate Hamming distance between target phrase and written phrase.
        """
        if len(self.target_phrase) != len(self.written_phrase):
            raise ValueError("Strings must be of equal length.")

        self.hamming = sum(c1 != c2 for c1, c2 in zip(self.target_phrase, self.written_phrase))
    
    def plot_hamming_distance(self, results_dir=None):
        """
        Plot the hamming distance as a function of time.
        """
        plt.title("Hamming distance")
        dist=0
        for i_character, character in enumerate(self.target_phrase):
            if character != self.written_phrase[i_character]:
                dist += 1
            plt.scatter(i_character, dist, c='blue')
        plt.xticks(ticks=np.arange(len(self.target_phrase)), labels=self.target_phrase)
        plt.xlabel("Number of written letters")
        plt.ylabel("Hamming distance")
        if results_dir is None:
            results_dir = "results"
        plt.savefig(f"{results_dir}/hamming_distance.png")
        plt.close()
    
    def calculate_distribution_distance(self):
        # l2_distance = np.linalg.norm(self.l_means_estimate - self.l_means_true, axis=1)
        KL_divergence = np.zeros(self.n_classes)
        for i_class in range(self.n_classes):
            KL_divergence[i_class] = 0.5 * (
                np.trace(
                    np.linalg.inv(self.l_covariances_estimate[i_class]) @ self.l_covariances_true[i_class]
                ) +
                np.dot(
                    np.dot(
                        (self.l_means_estimate[i_class] - self.l_means_true[i_class]),
                        np.linalg.inv(self.l_covariances_estimate[i_class])
                    ),
                    (self.l_means_estimate[i_class] - self.l_means_true[i_class])
                ) - self.dim +
                np.log(
                    np.linalg.det(self.l_covariances_estimate[i_class]) / np.linalg.det(self.l_covariances_true[i_class])
                )
            )
        return KL_divergence    
    
    def visualize_kl_divergence(self, kl_divergence, results_dir=None):
        """
        Plot KL divergence.
        """
        fig, axes = plt.subplots(11, 1)
        for i_class in range(self.n_classes):
            axes[i_class].scatter(range(len(self.written_phrase)), kl_divergence[:, i_class], label=f" Class {i_class}")
            axes[i_class].legend(loc='center left', bbox_to_anchor=(1, 0.5))  
            axes[i_class].set_xticks
        # set y ticks to be the same for all subplots
        # max_kl = np.max(kl_divergence)
        # for i_class in range(self.n_classes):
        #     if i_class == self.n_classes - 1:
        #         axes[i_class].set_yticks([0, max_kl])
        #     else:
        #         axes[i_class].set_yticks([max_kl])
        #     axes[i_class].set_ylim(0, max_kl)
            # axes[i_class].set_yticks([0, max_kl])
            # axes[i_class].set_yticks(np.arange(0, max_kl, max_kl/2))
        fig.supxlabel("# of written letters")
        fig.supylabel("KL Divergence")
        if results_dir is None:
            results_dir = "results"
        plt.savefig(f"{results_dir}/kl_divergence.png", bbox_inches='tight')
        plt.close()

    def visualize_entropy(self, entropy):
        """
        Visualize entropy.
        """
        plt.plot(entropy)
        plt.show()
        plt.close()

    def visualize_clusters(self, results_dir=None):
        """
        Visualize clusters.
        """
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan', 'magenta', 'yellow']
        fig, ax = plt.subplots()
        for i_inquiry in range(self.inquiry_length):
            if i_inquiry == 0:
                ax.scatter(self.l_means_estimate[i_inquiry, 0], self.l_means_estimate[i_inquiry, 1],
                       color="red", marker="x", label="Estimate")
            else:
                ax.scatter(self.l_means_estimate[i_inquiry, 0], self.l_means_estimate[i_inquiry, 1],
                       color="red", marker="x")
            eigenvalues, eigenvectors = np.linalg.eigh(self.l_covariances_estimate[i_inquiry])
            eigenvalues = 2.0 * np.sqrt(2.0 * eigenvalues) 
            ellipse_axis = eigenvectors[0] / np.linalg.norm(eigenvectors[0])
            angle = 180 * np.arctan(ellipse_axis[1] / ellipse_axis[0]) / np.pi 
            ellipse1 = Ellipse(self.l_means_estimate[i_inquiry], eigenvalues[0], eigenvalues[1],
                              angle=angle + 180.0, color=colors[i_inquiry], alpha=0.20)
            ax.add_artist(ellipse1)

        for i_inquiry in range(self.inquiry_length):
            if i_inquiry == 0:
                ax.scatter(self.l_means_true[i_inquiry, 0], self.l_means_true[i_inquiry, 1],
                       color="blue", marker="o", facecolors="none", label="True")
            else:
                ax.scatter(self.l_means_true[i_inquiry, 0], self.l_means_true[i_inquiry, 1],
                       color="blue", marker="o", facecolors="none")
            eigenvalues, eigenvectors = np.linalg.eigh(self.l_covariances_true[i_inquiry])
            eigenvalues = 2.0 * np.sqrt(2.0 * eigenvalues) 
            ellipse_axis = eigenvectors[0] / np.linalg.norm(eigenvectors[0])
            angle = 180 * np.arctan(ellipse_axis[1] / ellipse_axis[0]) / np.pi 
            ellipse2 = Ellipse(self.l_means_true[i_inquiry], eigenvalues[0], eigenvalues[1],
                              angle=angle + 180.0, color=colors[i_inquiry], alpha=0.10)
            ax.add_artist(ellipse2)
        plt.legend()
        # save figure in results_dir
        if results_dir is None:
            results_dir = "results"
        plt.savefig(f"{results_dir}/clusters_{len(self.written_phrase)}.png")
        plt.close()

    def visualize_training_samples(self, training_samples):
        """
        Visualize training samples.
        """
        fig, ax = plt.subplots()
        for i_class in range(self.n_classes):
            ax.plot(self.time_samples, training_samples[:, i_class, :].T, alpha=0.5, label=f"Class {i_class}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (uV)")
        plt.legend()
        plt.show()
        plt.close()

    def visualize_posterior(self, posterior):
        """
        Visualize posterior probability.
        """
        plt.scatter(np.arange(len(posterior)), posterior)
        plt.xticks(ticks=np.arange(len(posterior)), labels=self.alphabet)
        plt.show()
        plt.close()

    def visualize_nll(self, results_dir=None):
        plt.plot(range(len(self.nll)), self.nll)
        plt.xlabel("Iteration")
        plt.ylabel("Negative log likelihood")
        if results_dir is None:
            results_dir = "results"
        plt.savefig(f"{results_dir}/nll_{len(self.written_phrase):03d}_{self.written_phrase[-1]}.png")
        plt.close()

    def calculate_character_posterior(self, measurement, inquiry):
        """
        Calculate the posterior probability for each character given measurements.

        :return posterior list: Posterior probability for each character.
        """
        posterior = np.zeros_like(self.prob_alphabet)
        for i_character, character in enumerate(self.alphabet):
            if character in inquiry:
                i_inquiry = inquiry.index(character) + 1   # target location
            else:
                i_inquiry = 0   # non-target

            if np.isinf(self.l_covariances_estimate[i_inquiry]).any() or np.isnan(self.l_covariances_estimate[i_inquiry]).any():
                # print("Covariance matrix is singular.")
                pass
            else:
                posterior[i_character] = (
                    self.prob_alphabet[i_character] *
                    multivariate_normal.pdf(
                        measurement,
                        self.l_means_estimate[i_inquiry],
                        self.l_covariances_estimate[i_inquiry]
                        )
                )
        posterior /= sum(posterior)
        return posterior
    
    def calculate_p_theta(self):
        p_theta = multivariate_normal.logpdf(self.measurement,  self.n_means_estimate, self.n_covariances_estimate)
        return p_theta

    def p_x_given_theta_and_q(self, params, character_posterior, x_s, q_s):
        p_q = jnp.zeros(self.n_classes)
        p_x_given_theta = jnp.zeros(self.n_classes)
        p_theta_given_q = jnp.eye(self.n_classes)
        try:
            cov = self.generate_cov_matrix(params[1], params[2])
        except:
            cov = self.generate_cov_matrix(params[1].primal, params[2].primal)
        for i_label in range(self.n_classes):
            p_q = p_q.at[i_label].set(jnp.dot(character_posterior, q_s[i_label]))
            # p_x_given_theta = p_x_given_theta.at[i_label].set(jax.scipy.stats.multivariate_normal.pdf(x_s, params[0].primal[i_label], cov[i_label]))
            p_x_given_theta = p_x_given_theta.at[i_label].set(jax.scipy.stats.multivariate_normal.pdf(x_s, params[0][i_label], cov[i_label]))
        J = jnp.dot(jnp.dot(p_x_given_theta, p_theta_given_q), p_q)
        return J

    def negative_log_likelihood(self, params, character_posterior, l_measurements, Q_s):
        """
        Calculate negative log likelihood.
        """
        log_sum = 0
        for i_sequence, (measurement, q_s) in enumerate(zip(l_measurements, Q_s)):
            log_sum += jnp.log(self.p_x_given_theta_and_q(params, character_posterior, measurement, q_s))
        return (-log_sum).reshape(())   # ML update
        # return -log_sum - jnp.log(self.parameter_prior)  # MAP update
    
    def calculate_q_s(self, l_inquiries):
        """
        Generate Q_s matrix using l_inquiries information and the character posterior.
        Rows indicate the the label l=0,...L, columns are a binary vector indicating the presence of the character in each label
        """
        Q_s = np.zeros((len(l_inquiries), self.n_classes, len(self.alphabet)))
        for i_inquiry, inquiry in enumerate(l_inquiries):
            for i_character, character in enumerate(self.alphabet):
                for i_trial, trial in enumerate(inquiry):
                    if character == trial:
                        Q_s[i_inquiry, i_trial+1, i_character] = 1
                if character not in inquiry:
                    Q_s[i_inquiry, 0, i_character] = 1
        return Q_s

    def update_parameters(self, character_posterior, l_inquiries, l_measurements):
        """
        Update distribution paprameters.
        Calculate mean and covariance that maximizes the eq. 12:
        P(X_1, ..., X_s) = Product(Sum(P(X_i | Q_i) * P(Q_i_l) * P(Phi_l) for all l) for all i)
        P(Phi_l) is the prior probability of the lth cluster.
        P(Q_i_l) is the probability distribution for the ith inquiry given the lth cluster.
        P(X_i | Q_i) is the probability distribution for the ith inquiry given the inquiry.
        """
        params = [jnp.asarray(self.l_means_estimate), jnp.asarray(self.vec_a), jnp.asarray(self.vec_d)]
        optimizer = optax.sgd(learning_rate=self.stepsize)
        # optimizer = optax.adam(learning_rate=self.stepsize)
        # optimizer = optax.adamw(learning_rate=self.stepsize, weight_decay=1e-5)
        opt_state = optimizer.init(params)
        Q_s = self.calculate_q_s(l_inquiries)
        for iter in range(self.maxiter):
            # if iter % 50 == 0:
            #     print("Iteration:", iter)
            grads = jax.grad(self.negative_log_likelihood)(params, character_posterior, l_measurements, Q_s)
            updates, opt_state = optimizer.update(grads, opt_state)
            params_temp = optax.apply_updates(params, updates)
            # Check if updates are containing NaN or inf
            if np.isnan(params_temp[1]).any() or np.isinf(params_temp[1]).any():
                # print(f"NaN or inf in updates. Leaving optimization at iter={iter}.")
                break
            else:
                params = optax.apply_updates(params, updates)
                self.nll.append(self.negative_log_likelihood(params, character_posterior, l_measurements, Q_s))
        self.l_means_estimate, self.vec_a, self.vec_d = params
        self.l_covariances_estimate = self.generate_cov_matrix(self.vec_a, self.vec_d)

    def calibrate(self, n_samples_per_class=20, results_dir=None):
        """ Sample from the true distribution. Generate initial estimates for parameters."""
        samples_arr = np.zeros((self.n_electrodes, self.n_classes * n_samples_per_class, self.dim))
        labels_arr = np.zeros(self.n_classes * n_samples_per_class)
        for i_sample in range(n_samples_per_class):
            for i_class, (mean_, cov_) in enumerate(zip(self.l_means_true, self.l_covariances_true)):
                # Generate samples from the true distribution.
                ind_ =  i_sample * self.n_classes + i_class
                samples_arr[0, ind_] = scipy.stats.multivariate_normal.rvs(mean_, cov_)
                labels_arr[ind_] = i_class
       

        # fig, (ax1, ax2) = plt.subplots(1, 2)  
        # ax1.plot(self.time_samples, samples_arr[:,0,:].T, alpha=0.5)
        # ax1.set_xlabel("Time (s)")
        # ax1.set_ylabel("Amplitude (uV)")
        # ax2.plot(self.time_samples, samples_arr[:,1,:].T, alpha=0.5)
        # ax2.set_xlabel("Time (s)")
        # if results_dir is None:
        #     results_dir = "results"
        # plt.savefig(f"{results_dir}/Calibration_Samples.png")
        # plt.close()
        self.visualize_training_samples(samples_arr)     
        samples_centered = np.zeros((self.n_classes * n_samples_per_class, self.dim))
        for i_class in range(self.n_classes):
            self.l_means_estimate[i_class] = np.mean(samples_arr[0, labels_arr == i_class], axis=0)
            ind_ = i_class * n_samples_per_class
            for i_sample in range(n_samples_per_class):
                # Center the samples
                samples_centered[ind_ + i_sample] = samples_arr[0, labels_arr == i_class][i_sample] - self.l_means_estimate[i_class]

        for i_class in range(self.n_classes):    
            self.l_covariances_estimate[i_class] = np.cov(samples_centered, rowvar=False)

        # self.plot_means(self.l_means_estimate)
        # model = ModifiedRdaModel(k_folds=10, pca_n_components=self.dim, data_type="synthetic", n_classes=self.n_classes)
        # rda_means, rda_covs = model.fit(samples_arr, labels_arr)
        
        # Find the mean and covariance estimates after RDA
        # self.l_means_estimate = np.array(rda_means)
        # self.l_covariances_estimate = np.array(rda_covs)

        # Visually check the means and covariances
        for i_class in range(self.n_classes):
            plt.plot(range(len(self.l_means_estimate[i_class])), self.l_means_estimate[i_class], label=f"Class {i_class}")
        plt.legend()
        plt.title("RDA means")
        plt.show()
    

    def simulate(self, results_dir=None, update_toggle=True):
        # Show the initial clusters.
        self.visualize_clusters(results_dir)
        kl_divergence = []
        # print("Initial parameters")
        # print("Means:", self.l_means_estimate, "Covariances:", self.l_covariances_estimate)
        for i_character, character in enumerate(self.target_phrase):
            # Check if the probability of any character has exceeded the
            # threshold.
            print(f"Progress: {i_character / len(self.target_phrase):.3f}")
            self.prob_alphabet = self.prob_alphabet_prior.copy()
            self.entropies[i_character].append(
                self.calculate_entropy(self.prob_alphabet)
            )
            while all(self.prob_alphabet < self.threshold):
                # Generate inquiry.
                inquiry = self.generate_inquiry()
                self.inquiries[i_character].append(inquiry)


                # Generate measurement.
                # The following block checks in which position of the
                # inquiry the target character is located [1,...,10]. If it is not
                # shown in the inquiry then it is assigned the first index. [0]
                if character in inquiry:
                    character_position = inquiry.index(character) + 1
                else:
                    character_position = 0
                measurement = self.generate_measurement(character_position)
                self.measurements[i_character].append(measurement)
                
                # Calculate posterior given the measurement.
                posterior = self.calculate_character_posterior(measurement, inquiry)
                self.posteriors[i_character].append(posterior)
                self.entropies[i_character].append(
                    self.calculate_entropy(posterior)
                )
                
                # Update prior probability for the alphabet. 
                self.prob_alphabet = posterior.copy()            
            # Write character that has crossed threshold.
            self.written_phrase.append(
                alphabet[np.argmax(self.prob_alphabet)]
            )
            # self.visualize_entropy(self.entropies[i_character])
            # self.visualize_posterior(posterior)
            if update_toggle:
                self.update_parameters(posterior, self.inquiries[i_character], self.measurements[i_character])
            self.visualize_nll(results_dir)
            # print("Updated parameters")
            # print("Means:", self.l_means_estimate, "Covariances:", self.l_covariances_estimate)
            kl_divergence.append(self.calculate_distribution_distance())

        # Show the final clusters
        self.visualize_clusters(results_dir)
        kl_divergence = np.array(kl_divergence)
        self.visualize_kl_divergence(kl_divergence, results_dir)
        # print("Target phrase:", self.target_phrase)
        # print("Written phrase:", self.written_phrase)
        self.calculate_hamming_distance()
        self.plot_hamming_distance(results_dir)


if __name__ == "__main__":
    alphabet= 'abcdefghijklmnopqrstuvwxyz<_'
    target_phrase = "hello_world"
    # target_phrase = "earth day has always been a day that acknowledges our planet, " \
    # "which provides for us, and ways we can protect and preserve its beauty"
    target_phrase = target_phrase.replace(" ", "_")
    target_phrase = target_phrase.replace(".", "_")
    target_phrase = target_phrase.replace(",", "")

    n_alphabet = len(alphabet)
    prob_alphabet = 1 / n_alphabet * np.ones(n_alphabet)
    inquiry_length = 10
    seed_inquiry = 1960
    threshold = 0.7
    runID = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = "results/" + runID
    try:
        os.makedirs(results_dir)
    except FileExistsError:
        pass

    simulation = Simulation(target_phrase, alphabet, inquiry_length, prob_alphabet,
                            seed_inquiry, threshold)
    simulation.calibrate(n_samples_per_class=20, results_dir=results_dir)
    simulation.simulate(results_dir=results_dir, update_toggle=True)

    # Save as pickle
    with open(f"{results_dir}/simulation.pkl", "wb") as f:
        pickle.dump(simulation, f)

    # Save model attributes as text
    with open(f"{results_dir}/simulation_log.txt", "w") as f:
        f.write("Model Attributes:\n" + "-------------------\n")
        f.write("Optimizer: SGD\n")
        f.write("Max. iterations: " + str(simulation.maxiter) + "\n")
        f.write("Step size: " + str(simulation.stepsize) + "\n")
        f.write("Decision Threshold: " + str(simulation.threshold) + "\n" + "\n")
        f.write("Experiment Results:\n" + "-------------------\n")
        f.write("Experiment ID: " + runID + "\n")
        f.write("Target phrase: " + target_phrase + "\n")
        f.write("Data type: Synthetic\n")
        f.write("Written phrase: " + "".join(simulation.written_phrase) + "\n")
        f.write("Target phrase length: " + str(len(target_phrase)) + "\n")
        f.write("Hamming distance: " + str(simulation.hamming) + "\n")

