import torch


class GradientXInput:
    def __init__(self, model):
        self.model = model

    def attribute(self, inp, no_jacobian=False):
        """
        Computes the gradient times input attributions for the given input.

        Uses the torch jacobian function to compute the partial gradients of the model w.r.t. its parameters.
        Multiplies these gradients with the respective input values afterwards.no_jacibian indicates that the standard backward function of torch should be used instead.
        This works if the output of the model is a scalar.
        :param inp: tuple of inputs for the attributions
        :param no_jacobian: If the standard torch backward function should be used instead of the jacobian
        :return: attributions - tuple with the same structure as the jacobian of the model
        """

        num_inputs = len(inp)
        if no_jacobian:
            result = self.model.forward(*inp)
            result.backward()
            jacobian = [i.grad for i in inp]
        else:
            jacobian = torch.autograd.functional.jacobian(self.model, inp)

        result = []

        for i in range(num_inputs):
            gradxinput = torch.mul(jacobian[i], inp[i])
            result.append(gradxinput)

        return tuple(result)









