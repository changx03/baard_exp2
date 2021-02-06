import logging
import math
from typing import Optional, Union, TYPE_CHECKING

from torch.nn import Softmax
import numpy as np
from tqdm.auto import trange
import torch
from art.config import ART_NUMPY_DTYPE
from art.attacks.evasion.auto_projected_gradient_descent import AutoProjectedGradientDescent
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format, projection, random_sphere, is_probability, get_labels_np_array

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)

class AutoProjectedGradientDescentDetectors(AutoProjectedGradientDescent):
    """
    """
    attack_params = AutoProjectedGradientDescent.attack_params + [
        "detector",
        "detector_th",
        "beta",
        "clf_loss_multiplier",
    ]

    _predefined_losses = ["cross_entropy", 'difference_logits_ratio', "logits_difference"]

    def __init__(
        self,
        estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        detector: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        detector_th : int = 0.5,
        beta: int = 0.5,
        clf_loss_multiplier: int = 1,
        detector_clip_fun = None,
        norm: Union[int, float, str] = np.inf,
        eps: float = 0.3,
        eps_step: float = 0.1,
        max_iter: int = 100,
        targeted: bool = False,
        nb_random_init: int = 5,
        batch_size: int = 32,
        loss_type: Optional[str] = None,
        verbose: bool = True):
        """
        Create a :class:`.AutoProjectedGradientDescentDetectors` instance.

        :param estimator: A trained estimator.
        :param detector: A trained detector. Its prediction should be equal
        to 1 for the sample predicted as malicious and 0 for the ones
        predicted as benign.
        :param detector_th: Threshold to have a chosen number of false
        positives.
        :param beta: Constant which regulates the trade-off between the
        optimization of the classifier and the detector losses. In
        particular is the weight given to the detector's loss.
        :param clf_loss_multiplier: Constant for which the classifier loss
        will be multiplied.
        :param norm: The norm of the adversarial perturbation. Possible values: "inf", np.inf, 1 or 2.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param nb_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0
            starting at the original input.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param verbose: Show progress bars.
        """
        from art.estimators.classification import PyTorchClassifier

        self.beta = beta
        self.detector_th = detector_th
        self.clf_loss_multiplier = clf_loss_multiplier

        self.detector_clip_fun = detector_clip_fun

        if targeted is True:
            raise NotImplementedError("This attack so far do not works as a "
                                      "targeted attack. (the objective "
                                      "function and its gradient function "
                                      "need a little fix to make it work).")

        if isinstance(detector, PyTorchClassifier):
            import torch

            if detector.clip_values is not None:
                raise ValueError("The clip value of the detector cannot "
                                 "be different from None.")

            class detector_loss:
                """
                The detector loss is the detector score for the class 1
                - the detector threshold
                """

                def __init__(self):
                    self.reduction = "mean"

                def __call__(self, y_pred, y_true):  # type: ignore
                    """
                    y_pred are actually the logits.
                    y_true is actually unused.
                    """
                    if isinstance(y_pred, np.ndarray):
                        scores = torch.from_numpy(y_pred)
                    else:
                        scores = y_pred

                    # apply the softmax to have scores in 0 1
                    softmax_obj = Softmax(dim=1)
                    scores = softmax_obj(scores)

                    # consider the score assigned to the malicious class
                    scores = scores[:, 1]
                    scores = scores - detector_th

                    # create a vector of zeros
                    zero_vector = torch.zeros_like(scores)

                    # get the maximum values between scores - threshold and 0
                    scores = torch.max(scores, zero_vector)

                    if self.reduction == 'mean':
                        return torch.mean(scores)
                    else:
                        return scores

            self._det_loss_object = detector_loss()

            detector_apgd = PyTorchClassifier(
                model=detector.model,
                loss=self._det_loss_object,
                input_shape=detector.input_shape,
                nb_classes=detector.nb_classes,
                optimizer=None,
                channels_first=detector.channels_first,
                preprocessing_defences=detector.preprocessing_defences,
                postprocessing_defences=detector.postprocessing_defences,
                preprocessing=detector.preprocessing,
                device_type=detector._device,
            )

            self._det_loss_object = detector_loss()

        else:
            raise ValueError("The type of the detector classifier is not "
                             "supported.")

        self.detector = detector_apgd

        super().__init__(
                        estimator = estimator, norm = norm, eps=eps,
                        eps_step = eps_step, max_iter = max_iter,
                        targeted = targeted, nb_random_init = nb_random_init,
                        batch_size = batch_size, loss_type = loss_type,
                        verbose = verbose)

    #######################################################

    def _grad_normalization(self, grad):
        shape = grad.shape
        grad = grad.reshape((grad.shape[0], -1))
        grad_norms = np.linalg.norm(grad, axis=1, ord=2)
        grad_bigger_than_0 = grad_norms > 0
        grad_norms = np.atleast_2d(grad_norms).T
        grad[grad_bigger_than_0,:] = grad[grad_bigger_than_0,:] / grad_norms[grad_bigger_than_0]
        return grad.reshape(shape)

    def _cmpt_grad(self, x, y):

        # set grad equal to the classifier gradient
        grad= self.clf_loss_multiplier * \
                   self.estimator.loss_gradient(x, y) * \
                   (1 - 2 * int(self.targeted))

        # normalize the gradients
        grad = self._grad_normalization(grad)

        scores = self.estimator.predict(x)
        y_pred = np.argmax(scores, axis=1)
        y_true = np.argmax(y, axis=1)

        # todo: this work only for indiscriminate attack
        misclass = y_pred != y_true
        if misclass.any():
            x_misclass = x[misclass, :]
            y_misclass = y[misclass, :]

            # cmpt detector grad
            det_grads = self.beta * \
                           self.detector.loss_gradient(x_misclass, np.ones(
                               y_misclass.shape))  # grad wrt malicious
            # class.
            det_grads = self._grad_normalization(det_grads)

            grad[misclass] -= det_grads

        return grad

    def _cmpt_loss_func(self, x, y):

        # set the loss as the clf loss
        loss = self.estimator.loss(x=x, y=y,
                                         reduction="none")

        loss = self.clf_loss_multiplier * loss

        loss = np.array(loss)

        # print("abs clf loss: ", np.mean(loss))

        scores = self.estimator.predict(x)
        y_pred = np.argmax(scores, axis=1)
        y_true = np.argmax(y, axis=1)

        # todo: this work only for indiscriminate attack
        misclass = np.not_equal(y_pred, y_true)

        if misclass.any():
            misclass = y_pred != y_true
            x_misclass = x[misclass, :]
            y_misclass = y[misclass, :]
            loss[misclass] -=  self.beta * \
                           self.detector.loss(x=x_misclass, y=y_misclass,
                                              reduction="none")

        return np.mean(loss)

    ########################################################

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        mask = kwargs.get("mask")

        y = check_and_transform_label_format(y, self.estimator.nb_classes)

        if y is None:
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size)).astype(np.int32)

        x_adv = x.astype(ART_NUMPY_DTYPE)

        for _ in trange(max(1, self.nb_random_init), desc="AutoPGD - restart", disable=not self.verbose):

            # Get classifier scores
            estimator_y_scores = self.estimator.predict(x_adv)

            # get the detector prediction (1 means the sample is predicted
            # as malicious, 0 as benign).
            detector_scores = self.estimator.predict(x_adv)
            detector_pred = np.argmax(detector_scores, axis=1)

            # the element of sample_is_robust will be 0 if the sample is
            # classified as the attacker wants, 1 otherwise
            if self.targeted:
                sample_is_robust = np.argmax(estimator_y_scores, axis=1) != np.argmax(y, axis=1)
            elif not self.targeted:
                sample_is_robust = np.argmax(estimator_y_scores, axis=1) == np.argmax(y, axis=1)

            # 1 if the sample is still correct (not classified as the target
            # class) or detected as adversarial example by the detector
            sample_is_robust = np.logical_or(sample_is_robust, detector_pred)

            # stop the attack if all the samples are classified as the
            # attacker want: misclassied (classified as the target class)
            # and predicted by the detector as benign samples
            if np.sum(sample_is_robust) == 0:
                break

            x_robust = x_adv[sample_is_robust]
            y_robust = y[sample_is_robust]
            x_init = x[sample_is_robust]

            n = x_robust.shape[0]
            m = np.prod(x_robust.shape[1:]).item()
            random_perturbation = (
                random_sphere(n, m, self.eps, self.norm).reshape(x_robust.shape).astype(ART_NUMPY_DTYPE)
            )

            x_robust = x_robust + random_perturbation / 100.0

            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_robust = np.clip(x_robust, clip_min, clip_max)

            if self.detector_clip_fun is not None:
                x_robust = self.detector_clip_fun(x_robust, self.estimator)

            perturbation = projection(x_robust - x_init, self.eps, self.norm)
            x_robust = x_init + perturbation

            # Compute perturbation with implicit batching
            for batch_id in trange(
                int(np.ceil(x_robust.shape[0] / float(self.batch_size))),
                desc="AutoPGD - batch",
                leave=False,
                disable=not self.verbose,
            ):
                self.eta = 2 * self.eps_step
                batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
                x_k = x_robust[batch_index_1:batch_index_2].astype(ART_NUMPY_DTYPE)
                x_init_batch = x_init[batch_index_1:batch_index_2].astype(ART_NUMPY_DTYPE)
                y_batch = y_robust[batch_index_1:batch_index_2]

                p_0 = 0
                p_1 = 0.22
                W = [p_0, p_1]

                while True:
                    p_j_p_1 = W[-1] + max(W[-1] - W[-2] - 0.03, 0.06)
                    if p_j_p_1 > 1:
                        break
                    W.append(p_j_p_1)

                W = [math.ceil(p * self.max_iter) for p in W]

                eta = self.eps_step
                self.count_condition_1 = 0

                for k_iter in trange(self.max_iter, desc="AutoPGD - iteration", leave=False, disable=not self.verbose):

                    # Get perturbation, use small scalar to avoid division by 0
                    tol = 10e-8

                    # Get loss gradient wrt input; invert it if attack is
                    # targeted
                    grad = self._cmpt_grad(x_k, y_batch)

                    assert x_k.shape == grad.shape

                    perturbation = grad

                    if mask is not None:
                        perturbation = perturbation * (mask.astype(ART_NUMPY_DTYPE))

                    # Apply perturbation and clip
                    z_k_p_1 = x_k + eta * perturbation

                    if self.estimator.clip_values is not None:
                        clip_min, clip_max = self.estimator.clip_values
                        z_k_p_1 = np.clip(z_k_p_1, clip_min, clip_max)

                    if self.detector_clip_fun is not None:
                        z_k_p_1 = self.detector_clip_fun(z_k_p_1, self.estimator)

                    if k_iter == 0:
                        x_1 = z_k_p_1
                        perturbation = projection(x_1 - x_init_batch, self.eps, self.norm)
                        x_1 = x_init_batch + perturbation

                        f_0 = self._cmpt_loss_func(x_k, y_batch)

                        f_1 = self._cmpt_loss_func(x_1, y_batch)

                        self.eta_w_j_m_1 = eta
                        self.f_max_w_j_m_1 = f_0

                        if f_1 >= f_0:
                            self.f_max = f_1
                            self.x_max = x_1
                            self.x_max_m_1 = x_init_batch
                            self.count_condition_1 += 1
                        else:
                            self.f_max = f_0
                            self.x_max = x_k.copy()
                            self.x_max_m_1 = x_init_batch

                        # Settings for next iteration k
                        x_k_m_1 = x_k.copy()
                        x_k = x_1

                    else:
                        perturbation = projection(z_k_p_1 - x_init_batch, self.eps, self.norm)
                        z_k_p_1 = x_init_batch + perturbation

                        alpha = 0.75

                        x_k_p_1 = x_k + alpha * (z_k_p_1 - x_k) + (1 - alpha) * (x_k - x_k_m_1)

                        if self.estimator.clip_values is not None:
                            clip_min, clip_max = self.estimator.clip_values
                            x_k_p_1 = np.clip(x_k_p_1, clip_min, clip_max)

                        if self.detector_clip_fun is not None:
                            x_k_p_1 = self.detector_clip_fun(x_k_p_1, self.estimator)

                        perturbation = projection(x_k_p_1 - x_init_batch, self.eps, self.norm)
                        x_k_p_1 = x_init_batch + perturbation

                        f_k_p_1 = self._cmpt_loss_func(x_k_p_1, y_batch)

                        if f_k_p_1 > self.f_max:
                            self.count_condition_1 += 1
                            self.x_max = x_k_p_1
                            self.x_max_m_1 = x_k
                            self.f_max = f_k_p_1

                        if k_iter in W:

                            rho = 0.75

                            condition_1 = self.count_condition_1 < rho * (k_iter - W[W.index(k_iter) - 1])
                            condition_2 = self.eta_w_j_m_1 == eta and self.f_max_w_j_m_1 == self.f_max

                            if condition_1 or condition_2:
                                eta = eta / 2
                                x_k_m_1 = self.x_max_m_1
                                x_k = self.x_max
                            else:
                                x_k_m_1 = x_k
                                x_k = x_k_p_1.copy()

                            self.count_condition_1 = 0
                            self.eta_w_j_m_1 = eta
                            self.f_max_w_j_m_1 = self.f_max

                        else:
                            x_k_m_1 = x_k
                            x_k = x_k_p_1.copy()

                y_pred_adv_k = self.estimator.predict(x_k)

                # get the detector prediction.
                detector_scores_k = self.estimator.predict(x_k)
                detector_pred_k = np.argmax(detector_scores_k, axis=1)
                # invert the detector prediction
                inv_detector_pred_k = np.invert(detector_pred_k)

                # the element of sample_is_not_robust_k will be 1 if the
                # sample is classified as the attacker wants, 0 otherwise

                if self.targeted:
                    # invert makes the bitwise not.
                    sample_is_not_robust_k = np.invert(np.argmax(y_pred_adv_k, axis=1) != np.argmax(y_batch, axis=1))
                elif not self.targeted:
                    sample_is_not_robust_k = np.invert(np.argmax(y_pred_adv_k, axis=1) == np.argmax(y_batch, axis=1))

                # to be classified as the attacker want the samples should
                # be misclassified (classified as the target) and not
                # classified by the detector as a malicious sample
                sample_is_not_robust_k = np.logical_and(sample_is_not_robust_k,
                                                 # (1 if classified by the
                                                        # detector as benign)
                                                    inv_detector_pred_k)

                x_robust[batch_index_1:batch_index_2][sample_is_not_robust_k] = x_k[sample_is_not_robust_k]

            x_adv[sample_is_robust] = x_robust

        return x_adv

