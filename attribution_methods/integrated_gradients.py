import numpy as np
import torch

import xlp_utils


class IntegratedGradients:
    def __init__(self, model):
        self.model = model

    def attribute(self, inp, baseline=None, steps=50, no_jacobian=False, plexplain=False):
        """
        Produce attributions with the integrated gradients method.

        :param inp: Input value to compute the attributions for
        :param baseline: Baseline or None - The baseline has to have the same shape as the input or None, which
          results in a all zero baseline.
        :param steps: The number of steps to approximate the integral
        :param no_jacobian: If the jacobian function from torch should NOT be used to compute the attributions.
          It is default that the function is used, so this parameter is default False. However, in some cases
          (very large LPs), the jacobian can be memory inefficient.
        :param plexplain: If attributions for the plexplain program should be computed (in this case, the
          parameters have a different structure and require a different handling).
        :return: The attributions and the outputs for the different steps. The latter can be used for debugging.
        """
        # convert the torch tensors to numpy arrays
        inp = [i.detach().numpy() for i in inp]
        if baseline is not None:
            # convert the baseline to numpy arrays
            baseline = [i.detach().numpy() for i in baseline]
        attributions = self.integrated_gradients(self.model, inp, self.output_and_gradients, baseline, steps,
                                                 no_jacobian, plexplain)
        return attributions

    @staticmethod
    def integrated_gradients(
            model,
            inp,
            predictions_and_gradients,
            baseline,
            steps=50,
            no_jacobian=False,
            plexplain=False):
        """
        Computes integrated gradients for a given model.
        Adapted from
        https://arxiv.org/abs/1703.01365
        In addition to the integrated gradients tensor, the method also
        returns some additional debugging information for sanity checking
        the computation. See sanity_check_integrated_gradients for how this
        information is used.

        Access to the specific model is provided to the method via a
        'predictions_and_gradients' function provided as argument to this method.
        The function takes a batch of inputs and a label, and returns the
        predicted probabilities of the label for the provided inputs, along with
        gradients of the prediction with respect to the input. Such a function
        should be easy to create in most deep learning frameworks.

        Args:
          model: The pytorch model for which the gradients should be computed
          inp: The specific input for which integrated gradients must be computed.
          predictions_and_gradients: This is a function that provides access to the
            network's predictions and gradients. It takes the following
            arguments:
            - inputs: A batch of tensors of the same same shape as 'inp'. The first
                dimension is the batch dimension, and rest of the dimensions coincide
                with that of 'inp'.
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
          steps: [optional] Number of interpolation steps between the baseline
            and the input used in the integrated gradients computation. These
            steps along determine the integral approximation error. By default,
            steps is set to 50.

          no_jacobian: Whether or not the jacobian function from torch should be used to compute
            the attributions.
          plexplain: If the attributions should be computed for the plexplain model. This model
            has different input and output shapes, requiring a different process of the data.
        Returns:
          integrated_gradients: The integrated_gradients of the prediction for the
            provided prediction label to the input. It has the same shape as that of
            the input.
          outputs: The model output at each interpolation step. This can be used as a sanity
            check and for debugging.

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
        outputs, grads = predictions_and_gradients(scaled_inputs, model,
                                                   no_jacobian, plexplain)
            # shapes: <steps+1>, <steps+1, inp.shape>

        # standard numpy operations have to be used on a lower level of the lists.
        # grads = (grads[:-1] + grads[1:]) / 2.0
        # avg_grads = np.average(grads, axis=0)

        accumulated_grads = []
        for i in grads[0]:
            accumulated_grads.append(np.zeros(i.shape))
        for i in range(len(grads) - 1):  # iterate over all full gradients
            for j in range(len(grads[i])):  # iterate over all partial gradients within one full gradient
                accumulated_grads[j] += (grads[i][j] + grads[i + 1][j]) / 2.0

        # average the accumulated gradients
        for i in range(len(accumulated_grads)):
            accumulated_grads[i] /= (len(grads) - 1.)

        avg_grads = accumulated_grads

        # only for plexplain:
        if plexplain:
            diff_inp_base = np.array([inp[i] - baseline[i] for i in range(len(inp))])
            integrated_gradients = [avg_grads[i] * diff_inp_base for i in range(len(avg_grads))]
        else:
            integrated_gradients = [(inp[i] - baseline[i]) * avg_grads[i] for i in range(len(inp))]  # shape: <inp.shape>

        return integrated_gradients, outputs

    @staticmethod
    def output_and_gradients(inputs, model, no_jacobian, plexplain):
        """
        predictions_and_gradients: This is a function that provides access to the
            network's predictions and gradients. It takes the following
            arguments:
            - inputs: A batch of tensors of the same same shape as 'inp'. The first
                dimension is the batch dimension, and rest of the dimensions coincide
                with that of 'inp'.
            - target_label_index: The index of the target class for which gradients
              must be obtained.
            - no_jacobian: Whether or not the jacobian function of torch should be used.
            - plexplain: If the model is a plexplain problem - requiring a different
              handling of the data.
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
            if plexplain:
                inp = [torch.tensor(inp, requires_grad=True)]
            else:
                inp = [torch.tensor(i, requires_grad=True) for i in inp]

            output = model.forward(*inp)  # requires the elements of inp as individual parameters

            if no_jacobian:
                output.backward()
                jac = [i.grad for i in inp]
            else:
                if plexplain:
                    jac = torch.autograd.functional.jacobian(model, *inp)
                else:
                    jac = torch.autograd.functional.jacobian(model, tuple(inp))

            output = xlp_utils.detach_tuple(output)
            outputs.append(output)
            gradient = xlp_utils.detach_tuple(jac)
            gradients.append(gradient)

        return outputs, gradients
