import numpy as np
import torch

import ex_lp_utils


class IntegratedGradients:
    def __init__(self, model):
        self.model = model

    def attribute(self, inp, baseline=None, steps=50, no_jacobian=False):
        """
        Produce attributions with the integrated gradients method.
        :param inp: Input value to compute the attributions for
        :param baseline: Baseline or None - The baseline has to have the same shape as the input or None, which results
            in a all zero baseline.
        :param steps: The number of steps to approximate the integral
        :return: The attributions and the outputs for the different steps
        """
        # convert the torch tensors to numpy arrays
        inp = [i.detach().numpy() for i in inp]
        if baseline is not None:
            # convert the baseline to numpy arrays
            baseline = [i.detach().numpy() for i in baseline]
        attributions = self.integrated_gradients(self.model, inp, self.output_and_gradients, baseline, steps, no_jacobian)
        return attributions

    @staticmethod
    def integrated_gradients(
            model,
            inp,
            predictions_and_gradients,
            baseline,
            steps=50,
            no_jacobian=False):
        """Computes integrated gradients for a given network and prediction label.
        Integrated gradients is a technique for attributing a deep network's
        prediction to its input features. It was introduced by:
        https://arxiv.org/abs/1703.01365
        In addition to the integrated gradients tensor, the method also
        returns some additional debugging information for sanity checking
        the computation. See sanity_check_integrated_gradients for how this
        information is used.

        This method only applies to classification networks, i.e., networks
        that predict a probability distribution across two or more class labels.

        Access to the specific network is provided to the method via a
        'predictions_and_gradients' function provided as argument to this method.
        The function takes a batch of inputs and a label, and returns the
        predicted probabilities of the label for the provided inputs, along with
        gradients of the prediction with respect to the input. Such a function
        should be easy to create in most deep learning frameworks.

        Args:
          model: The pytorch model for which the gradients should be computed
          inp: The specific input for which integrated gradients must be computed.
          target_label_index: Index of the target class for which integrated gradients
            must be computed.
          predictions_and_gradients: This is a function that provides access to the
            network's predictions and gradients. It takes the following
            arguments:
            - inputs: A batch of tensors of the same same shape as 'inp'. The first
                dimension is the batch dimension, and rest of the dimensions coincide
                with that of 'inp'.
            - target_label_index: The index of the target class for which gradients
              must be obtained.
            and returns:
            - predictions: Predicted probability distribution across all classes
                for each input. It has shape <batch, num_classes> where 'batch' is the
                number of inputs and num_classes is the number of classes for the model.
            - gradients: Gradients of the prediction for the target class (denoted by
                target_label_index) with respect to the inputs. It has the same shape
                as 'inputs'.
          baseline: [optional] The baseline input used in the integrated
            gradients computation. If None (default), the all zero tensor with
            the same shape as the input (i.e., 0*input) is used as the baseline.
            The provided baseline and input must have the same shape.
          steps: [optional] Number of intepolation steps between the baseline
            and the input used in the integrated gradients computation. These
            steps along determine the integral approximation error. By default,
            steps is set to 50.
        Returns:
          integrated_gradients: The integrated_gradients of the prediction for the
            provided prediction label to the input. It has the same shape as that of
            the input.

          The following output is meant to provide debug information for sanity
          checking the integrated gradients computation.
          See also: sanity_check_integrated_gradients
          prediction_trend: The predicted probability distribution across all classes
            for the various (scaled) inputs considered in computing integrated gradients.
            It has shape <steps, num_classes> where 'steps' is the number of integrated
            gradient steps and 'num_classes' is the number of target classes for the
            model.
        """

        if baseline is None:
            baseline = [np.zeros(i.shape) for i in inp]
        for i in range(len(inp)):
            assert inp[i].shape == baseline[i].shape

        # Scale input and compute gradients.
        scaled_inputs = []
        for i in range(0, steps + 1):
            scaled_input = []
            for j in range(len(inp)):
                scaled_input.append(baseline[j] + (float(i) / steps) * (inp[j] - baseline[j]))
            scaled_inputs.append(scaled_input)

        # scaled_inputs = [baseline + (float(i) / steps) * (inp - baseline) for i in range(0, steps + 1)]
        outputs, grads = predictions_and_gradients(scaled_inputs, model, no_jacobian)  # shapes: <steps+1>, <steps+1, inp.shape>


        # Use trapezoidal rule to approximate the integral.
        # See Section 4 of the following paper for an accuracy comparison between
        # left, right, and trapezoidal IG approximations:
        # "Computing Linear Restrictions of Neural Networks", Matthew Sotoudeh, Aditya V. Thakur
        # https://arxiv.org/abs/1908.06214

        # standard numpy operations have to be used on a lower level of the lists.
        # grads = (grads[:-1] + grads[1:]) / 2.0
        # avg_grads = np.average(grads, axis=0)

        accumulated_grads = []
        for i in grads[0]:
            accumulated_grads.append(np.zeros(i.shape))
        for i in range(len(grads) - 1):         # iterate over all full gradients
            for j in range(len(grads[i])):  # iterate over all partial gradients within one full gradient
                accumulated_grads[j] += (grads[i][j] + grads[i+1][j]) / 2.0

        # average the accumulated gradients
        for i in range(len(accumulated_grads)):
            accumulated_grads[i] /= (len(grads) - 1.)

        avg_grads = accumulated_grads


        integrated_gradients = [(inp[i] - baseline[i]) * avg_grads[i] for i in range(len(inp))]  # shape: <inp.shape>
        # only for plexplain:
        # diff_inp_base = np.array([inp[i] - baseline[i] for i in range(len(inp))])
        # integrated_gradients = [avg_grads[i] * diff_inp_base for i in range(len(avg_grads))]
        return integrated_gradients, outputs

    @staticmethod
    def output_and_gradients(inputs, model, no_jacobian):
        """
        predictions_and_gradients: This is a function that provides access to the
            network's predictions and gradients. It takes the following
            arguments:
            - inputs: A batch of tensors of the same same shape as 'inp'. The first
                dimension is the batch dimension, and rest of the dimensions coincide
                with that of 'inp'.
            - target_label_index: The index of the target class for which gradients
              must be obtained.
            and returns:
            - predictions: Predicted probability distribution across all classes
                for each input. It has shape <batch, num_classes> where 'batch' is the
                number of inputs and num_classes is the number of classes for the model.
            - gradients: Gradients of the prediction for the target class (denoted by
                target_label_index) with respect to the inputs. It has the same shape
                as 'inputs'.
        """
        gradients = []
        outputs = []
        for inp in inputs:
            inp = [torch.tensor(i, requires_grad=True) for i in inp]
            # only for plexplain:
            # inp = [torch.tensor(inp, requires_grad=True)]

            output = model.forward(*inp)            # requires the elements of inp as individual parameters

            if no_jacobian:
                output.backward()
                jac = [i.grad for i in inp]
            else:
                jac = torch.autograd.functional.jacobian(model, tuple(inp))
                # only for plexplain:
                # jac = torch.autograd.functional.jacobian(model, *inp)

            output = ex_lp_utils.detach_tuple(output)
            outputs.append(output)
            gradient = ex_lp_utils.detach_tuple(jac)
            gradients.append(gradient)


        # it is not possible to handle all gradients as a numpy array, because the individual gradients have
        # different shapes
        # gradients = np.array(gradients, dtype=object)

        return outputs, gradients
