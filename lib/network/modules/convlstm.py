import torch.nn as nn
from torch.autograd import Variable
import torch
import functools
from inplace_abn.bn import InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

class leaf_ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(leaf_ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Sequential(nn.Conv2d(in_channels=self.input_dim,
                              out_channels=3 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias), BatchNorm2d(3 * self.hidden_dim))

    def forward(self, input_tensor):

        combined_conv = self.conv(input_tensor)

        cc_i, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class plus_tree_ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(plus_tree_ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv_input = nn.Sequential(nn.Conv2d(in_channels=self.input_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias), BatchNorm2d(4 * self.hidden_dim))

        self.conv_h = nn.Sequential(nn.Conv2d(in_channels=self.hidden_dim,
                                            out_channels=3 * self.hidden_dim,
                                            kernel_size=self.kernel_size,
                                            padding=self.padding,
                                            bias=self.bias), BatchNorm2d(3 * self.hidden_dim))

        self.conv_hf = nn.Sequential(nn.Conv2d(in_channels=self.hidden_dim,
                                            out_channels=self.hidden_dim,
                                            kernel_size=self.kernel_size,
                                            padding=self.padding,
                                            bias=self.bias), BatchNorm2d(self.hidden_dim))

    def forward(self, input_tensor, child_cur_state_list):
        h_cur_list, c_cur_list = child_cur_state_list

        # combined_h = h_cur_list[0]
        # for i in range(1, len(h_cur_list)):
        #     combined_h = combined_h + h_cur_list[i]
        combined_h = torch.sum(torch.stack(h_cur_list, dim=1), dim=1)

        # combined_h = torch.max(torch.stack(h_cur_list, dim=1), dim=1)[0]

        combined_conv_input = self.conv_input(input_tensor)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv_input, self.hidden_dim, dim=1)
        combined_conv_h = self.conv_h(combined_h)
        ch_i, ch_o, ch_g = torch.split(combined_conv_h, self.hidden_dim, dim=1)

        ch_f_list = [self.conv_hf(h_cur_list[i]) for i in range(0, len(h_cur_list)-1)]
        i = torch.sigmoid(cc_i+ch_i)
        f_list = [torch.sigmoid(cc_f+ch_f) for ch_f in ch_f_list]
        o = torch.sigmoid(cc_o+ch_o)
        g = torch.tanh(cc_g+ch_g)


        fc_list = [f_list[i] * c_cur_list[i] for i in range(0, len(h_cur_list)-1)]
        combined_fc = fc_list[0]
        for i in range(1, len(fc_list)):
            combined_fc = combined_fc + fc_list[i]
        c_next = combined_fc+ i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class max_tree_ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(max_tree_ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv_input = nn.Sequential(nn.Conv2d(in_channels=self.input_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias), BatchNorm2d(4 * self.hidden_dim))

        self.conv_h = nn.Sequential(nn.Conv2d(in_channels=self.hidden_dim,
                                            out_channels=3 * self.hidden_dim,
                                            kernel_size=self.kernel_size,
                                            padding=self.padding,
                                            bias=self.bias), BatchNorm2d(3 * self.hidden_dim))

        self.conv_hf = nn.Sequential(nn.Conv2d(in_channels=self.hidden_dim,
                                            out_channels=self.hidden_dim,
                                            kernel_size=self.kernel_size,
                                            padding=self.padding,
                                            bias=self.bias), BatchNorm2d(self.hidden_dim))

    def forward(self, input_tensor, child_cur_state_list):
        h_cur_list, c_cur_list = child_cur_state_list

        # combined_h = h_cur_list[0]
        # for i in range(1, len(h_cur_list)):
        #     combined_h = combined_h + h_cur_list[i]
        # combined_h = torch.sum(torch.stack(h_cur_list, dim=1), dim=1)
        combined_h = torch.max(torch.stack(h_cur_list, dim=1), dim=1)[0]

        combined_conv_input = self.conv_input(input_tensor)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv_input, self.hidden_dim, dim=1)
        combined_conv_h = self.conv_h(combined_h)
        ch_i, ch_o, ch_g = torch.split(combined_conv_h, self.hidden_dim, dim=1)

        ch_f_list = [self.conv_hf(h_cur_list[i]) for i in range(0, len(h_cur_list)-1)]
        i = torch.sigmoid(cc_i+ch_i)
        f_list = [torch.sigmoid(cc_f+ch_f) for ch_f in ch_f_list]
        o = torch.sigmoid(cc_o+ch_o)
        g = torch.tanh(cc_g+ch_g)


        fc_list = [f_list[i] * c_cur_list[i] for i in range(0, len(h_cur_list)-1)]
        combined_fc = fc_list[0]
        for i in range(1, len(fc_list)):
            combined_fc = combined_fc + fc_list[i]
        c_next = combined_fc+ i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class tree_ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(tree_ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv_input = nn.Sequential(nn.Conv2d(in_channels=self.input_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias), BatchNorm2d(4 * self.hidden_dim))

        self.conv_h = nn.Sequential(nn.Conv2d(in_channels=self.hidden_dim,
                                            out_channels=3 * self.hidden_dim,
                                            kernel_size=self.kernel_size,
                                            padding=self.padding,
                                            bias=self.bias), BatchNorm2d(3 * self.hidden_dim))

        self.conv_hf = nn.Sequential(nn.Conv2d(in_channels=self.hidden_dim,
                                            out_channels=self.hidden_dim,
                                            kernel_size=self.kernel_size,
                                            padding=self.padding,
                                            bias=self.bias), BatchNorm2d(self.hidden_dim))

    def forward(self, input_tensor, child_cur_state_list):
        h_cur_list, c_cur_list = child_cur_state_list

        combined_h = h_cur_list[0]
        for i in range(1, len(h_cur_list)):
            combined_h = combined_h + h_cur_list[i]

        combined_conv_input = self.conv_input(input_tensor)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv_input, self.hidden_dim, dim=1)
        combined_conv_h = self.conv_h(combined_h)
        ch_i, ch_o, ch_g = torch.split(combined_conv_h, self.hidden_dim, dim=1)

        ch_f_list = [self.conv_hf(h) for h in h_cur_list]
        i = torch.sigmoid(cc_i+ch_i)
        f_list = [torch.sigmoid(cc_f+ch_f) for ch_f in ch_f_list]
        o = torch.sigmoid(cc_o+ch_o)
        g = torch.tanh(cc_g+ch_g)


        fc_list = [f_list[i] * c_cur_list[i] for i in range(0, len(h_cur_list))]
        combined_fc = fc_list[0]
        for i in range(1, len(fc_list)):
            combined_fc = combined_fc + fc_list[i]
        c_next = combined_fc+ i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class Seq_ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(Seq_ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):

                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
