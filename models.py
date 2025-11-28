import nn
from backend import PerceptronDataset, RegressionDataset, DigitClassificationDataset


class PerceptronModel(object):
    def __init__(self, dimensions: int) -> None:
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self) -> nn.Parameter:
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"

        return nn.DotProduct(x, self.w)

    def get_prediction(self, x: nn.Constant) -> int:
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"

        score = nn.as_scalar(self.run(x))
        if score >= 0:
            return 1
        else:
            return -1

    def train(self, dataset: PerceptronDataset) -> None:
        """
        Train the perceptron until convergence.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"

        converged = False

        while not converged:
            converged = True

            for x, y in dataset.iterate_once(1):
                y_label = nn.as_scalar(y)
                y_predite = self.get_prediction(x)

                if y_predite != y_label:
                    # Mise a jour des poids du modele
                    self.w.update(direction = x, multiplier = y_label)
                    converged = False


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"

        dimension_des_couches_cachees = 150

        self.w1 = nn.Parameter(1, dimension_des_couches_cachees)
        self.b1 = nn.Parameter(1, dimension_des_couches_cachees)

        self.w2 = nn.Parameter(dimension_des_couches_cachees, dimension_des_couches_cachees)
        self.b2 = nn.Parameter(1, dimension_des_couches_cachees)

        self.w3 = nn.Parameter(dimension_des_couches_cachees, 1)
        self.b3 = nn.Parameter(1, 1)


    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"

        # Calcul intermediaire et sortie du neurone
        z1 = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        a1 = nn.ReLU(z1)

        z2 = nn.AddBias(nn.Linear(a1, self.w2), self.b2)
        a2 = nn.ReLU(z2)

        return nn.AddBias(nn.Linear(a2, self.w3), self.b3)


    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"

        y_predicted = self.run(x)
        return nn.SquareLoss(y_predicted, y)


    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"

        taux_apprentissage = 0.05
        taille_des_mini_batchs = 100
        taux_de_perte = 0.015

        while True:
            for x, y in dataset.iterate_once(taille_des_mini_batchs):
                pertes = self.get_loss(x, y)

                # Gradient de chaque parametre (backpropagation)
                grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3 = nn.gradients(pertes,[self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
                self.w1.update(grad_w1, -taux_apprentissage)
                self.b1.update(grad_b1, -taux_apprentissage)
                self.w2.update(grad_w2, -taux_apprentissage)
                self.b2.update(grad_b2, -taux_apprentissage)
                self.w3.update(grad_w3, -taux_apprentissage)
                self.b3.update(grad_b3, -taux_apprentissage)

            if nn.as_scalar(pertes) < taux_de_perte:
                break


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self) -> None:
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

        hidden = 150

        # Parametres couche 1
        self.w1 = nn.Parameter(784, hidden)
        self.b1 = nn.Parameter(1, hidden)

        # Parametres couche 2
        self.w2 = nn.Parameter(hidden, hidden)
        self.b2 = nn.Parameter(1, hidden)

        # Parametres couche 3
        self.w3 = nn.Parameter(hidden, 10)
        self.b3 = nn.Parameter(1, 10)

        self.parameters = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        # Couche 1
        z1 = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        h1 = nn.ReLU(z1)

        # Couche 2 
        z2 = nn.AddBias(nn.Linear(h1, self.w2), self.b2)
        h2 = nn.ReLU(z2)

        logits = nn.AddBias(nn.Linear(h2, self.w3), self.b3)
        return logits

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        logits = self.run(x)
        return nn.SoftmaxLoss(logits, y)

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        batch_size = 200
        learning_r = 0.08
        check_every = 200

        # genere les batchs jusqua ce que le seuil est verifier
        i = 0
        for x, y in dataset.iterate_forever(batch_size):
            i += 1
            loss = self.get_loss(x, y)
            grads = nn.gradients(loss, self.parameters)

            for p, g in zip(self.parameters, grads):
                p.update(g, -learning_r)

            # verifies la validation tous les 200 batchs pour run plus vite
            if i % check_every == 0:
                if dataset.get_validation_accuracy() >= 0.975:
                    break

