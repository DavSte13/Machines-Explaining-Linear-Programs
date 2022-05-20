import torch
import numpy as np


class GrangerCausal:
    def __init__(self, model, reduced_model, eval_function, reduce_rows, max_flow=False):
        """
        model is the normal model for which the attributions should be computed.
        reduced_model is the model which accepts parameters of reduced size (either one row or column less)
        reduce_rows specifies if the reduced model has reduced rows (true) or reduced columns (false)
        eval_function specifies the type of function for which the problem is evaluated (optimal solution 'opt'
        or cost function 'cost')
        if max_flow=True, the dimension of the parameter b is also reduced by one (necessary because iof the problem
        formulation)
        """
        self.model = model
        self.reduced_model = reduced_model
        self.eval_func = eval_function
        self.reduce_rows = reduce_rows
        if max_flow and reduce_rows:
            ValueError("The maximum flow problem is not supported with reduce_rows.")
        self.max_flow = max_flow

    def attribute(self, inp):
        """
        Compute the granger causal attributions for the input.
        inp is the input of the model: (a, b, c)
        """
        a, b, c = inp
        res = self.model.forward(a, b, c)

        if self.reduce_rows:
            # convert a and b to numpy arrays and split them into individual rows
            a, b = a.detach().numpy(), b.detach().numpy()
            num_rows = a.shape[0]
            rows_a = np.split(a, num_rows, 0)
            rows_b = np.split(b, num_rows, 0)

            attributions = []

            for i in range(num_rows):
                # combine all rows of a and b except the i-th one
                short_a = torch.tensor(np.vstack(np.delete(rows_a, i, 0)))
                short_b = torch.tensor(np.vstack(np.delete(rows_b, i, 0))[0])

                # compute the result of the reduced model
                short_res = self.reduced_model.forward(short_a, short_b, c)

                attributions.append(res - short_res)

        else:
            # convert a and c to numpy arrays and split them into individual columns
            a, c = a.detach().numpy(), c.detach().numpy()
            num_cols = a.shape[1]
            cols_a = np.split(a, num_cols, 1)
            cols_c = np.split(c, num_cols, 0)

            attributions = []

            if self.max_flow:
                b = b.detach().numpy()
                cols_b = np.split(b, num_cols, 0)

            for i in range(num_cols):
                # combine all columns of a and b except the i-th one
                short_a = torch.tensor(np.hstack(np.delete(cols_a, i, 0)))
                short_c = torch.tensor(np.hstack(np.delete(cols_c, i, 0)))

                if self.max_flow:
                    short_b = torch.tensor(np.hstack(np.delete(cols_b, i, 0)))

                # compute the result of the reduced model
                if self.max_flow:
                    short_res = self.reduced_model.forward(short_a, short_b, short_c)
                else:
                    short_res = self.reduced_model.forward(short_a, b, short_c)

                if self.eval_func == 'cost':
                    attributions.append(res - short_res)
                else:
                    # flatten the results (in case they are more than one-dimensional
                    short_res = torch.flatten(short_res)
                    res = torch.flatten(res)
                    # handle the case where the attributions are calculated for the optimal solution:
                    tmp = torch.cat((short_res[:i], torch.tensor([0]), short_res[i:]))
                    tmp = res - tmp
                    tmp[i] = float('Inf')
                    attributions.append(tmp)

        return attributions
