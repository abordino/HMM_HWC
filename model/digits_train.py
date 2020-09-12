import csv
import logging
import pathlib
import pickle
import numpy as np
from model import HMM_class


class DigitsHmm:

    def __init__(self, zinga):
        """
            Initialize the model of digits.

            Args_1:
                 zinga (dict): dictionary of the model.
        """
        self.zinga = zinga

    def train_model(self, n, m):
        """
            Train the model using the baum_welch function.

            Args_1:
                n (int): number of hidden states.
            Args_2:
                m (int): length of the block of each single observation (# signals = 2 ** m).
        """
        assert m in [1, 2, 4, 7]
        self.zinga["indexes"] = [n, m]
        path = pathlib.Path("data/data_backup/" + str(m) + "x" + str(int(784 / m)))

        for i in range(10):
            with open(path.__str__() + "/train_" + str(i) + ".csv", "r") as p:
                train_list = [[int(x) for x in rec[1:]] for rec in csv.reader(p, delimiter=',')]

            rand_A = np.random.rand(n, n)
            init_A = rand_A / rand_A.sum(axis=1)[:, None]
            rand_B = np.random.rand(n, 2 ** m)
            init_B = rand_B / rand_B.sum(axis=1)[:, None]
            rand_pi = np.random.rand(1, n)
            init_pi = rand_pi / np.sum(rand_pi)

            hmm = HMM_class.HMM(init_A, init_B, init_pi)
            self.zinga[i] = hmm.baum_welch_train(train_list, n)

        print(self.zinga)
        with open("zinga.pickle", "wb") as zinga_file:
            pickle.dump(self.zinga, zinga_file)

    def load_digit_model(self):
        """
            Load the model with attribute n, m.
        """

        path = pathlib.Path("zinga.pickle")

        if path.exists():
            try:
                with open(path, "rb") as zinga_file:
                    zinga_dict = pickle.load(zinga_file)
                    print(zinga_dict)
                    self.zinga = zinga_dict

            except (OSError, IOError) as e:
                logging.exception("LOAD", e)
        else:
            print("No model found. Train it first.")
            dictionary = {}
            with open(path, "wb") as zinga_file:
                pickle.dump(dictionary, zinga_file)

    def model_predict(self, test_observation):
        """
            Make prediction using the model zinga.

            Args_1:
                test_observation (list): an observation to classify.
        """
        best_score = - float('inf')
        predicted_digit = None

        assert self.zinga != {}
        try:
            for i in range(10):
                hmm_i = HMM_class.HMM(self.zinga[i][0], self.zinga[i][1], self.zinga[i][2])
                score_new = hmm_i.score(test_observation)

                if score_new > best_score:
                    best_score = score_new
                    predicted_digit = i

            return predicted_digit

        except Exception as e:
            logging.exception("PREDICT: ", e)

    def test_model(self):
        """
            Test the zinga model on the observation dataset.
        """
        total_correct = 0
        total = 0

        assert self.zinga != {}
        try:
            m = self.zinga["indexes"][0]

            for i in range(10):
                correct = 0
                path = pathlib.Path("data/data_backup/" + str(m) + "x" + str(int(784 / m)) + "/test_" + str(i) + ".csv")

                with open(path, "r") as p:
                    data_test = [[int(x) for x in rec] for rec in csv.reader(p, delimiter=',')]

                for j in range(len(data_test)):
                    if self.model_predict(data_test[j][1:]) == data_test[j][0]:
                        correct += 1
                        total_correct += 1
                        total += 1
                    else:
                        total += 1
                print("accuracy_" + str(i) + " = ", correct / len(data_test))

            print("Total accuracy = ", total_correct / total)

        except Exception as e:
            logging.exception("TEST: ", e)


if __name__ == "__main__":

    def see_model():
        """
            See the model with attribute n, m.
        """
        path = pathlib.Path("zinga.pickle")
        if path.exists():
            with open(path, "rb") as zinga_file:
                zinga_dict = pickle.load(zinga_file)

            for i in range(10):
                print(i, "\nA:", zinga_dict[i][0], "\nB:", zinga_dict[i][1], "\npi:", zinga_dict[i][2])
        else:
            print("No model found")


    n = int(input("Select number of hidden states n -> "))
    m = int(input("Select length of blocks m (# signals 2 ** m) -> "))
    model = DigitsHmm({})
    model.train_model(n, m)
    model.test_model()
    see_model()
