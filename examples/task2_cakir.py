#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#
#     DCASE 2017
#     Task 2: Rare Sound Event Detection
#     A custom learner where you can create FNNs, CNNs, RNNs and CRNNs of any size and depth
#     Code used in the paper:
#     Convolutional Recurrent Neural Networks for Rare Sound Event Detection (Emre Cakir, Tuomas Virtanen)
#     http://www.cs.tut.fi/sgn/arg/dcase2017/documents/challenge_technical_reports/DCASE2017_Cakir_104.pdf
#
#     ---------------------------------------------
#         Tampere University of Technology / Audio Research Group
#         Author:  Emre Cakir ( emre.cakir@tut.fi )
#
#
#
from __future__ import print_function, absolute_import
import sys
import os

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
import numpy
import textwrap
import platform

from dcase_framework.application_core import BinarySoundEventAppCore
from dcase_framework.parameters import ParameterContainer
from dcase_framework.utils import *
from dcase_framework.learners import EventDetectorKerasSequential



__version_info__ = ('1', '0', '0')
__version__ = '.'.join(__version_info__)

class MultifunctionalDeepLearningSequential(EventDetectorKerasSequential):
    """
    A sequential multifunctional model, from which DNN, CNN, RNN and CRNN models can be created.
    """

    def __init__(self, *args, **kwargs):
        super(MultifunctionalDeepLearningSequential, self).__init__(*args, **kwargs)
        self.method = 'multifunctionalDNN_seq'

    def create_model(self, input_shape):
        from keras.layers import MaxPooling1D, Convolution2D, MaxPooling2D, Input, RepeatVector
        from keras.layers.advanced_activations import LeakyReLU
        from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape, Permute
        from keras.layers.recurrent import LSTM, GRU
        from keras.layers.normalization import BatchNormalization
        from keras.models import Model
        from keras.regularizers import L1L2
        from keras.layers.wrappers import TimeDistributed
        from keras.legacy.layers import MaxoutDense

        learner_params = self.learner_params

        # Read general model parameters.
        batch_size = learner_params['training']['batch_size']
        dropout_flag = learner_params['general']['dropout_flag']
        dropout_rate = learner_params['general']['dropout_rate']
        l1_weight_norm = learner_params['general']['l1_weight_norm']
        l2_weight_norm = learner_params['general']['l2_weight_norm']
        output_act = learner_params['general']['output_act']

        # Read feed-forward layer parameters.
        nb_fnn_hidden_units = learner_params['fnn_params']['nb_fnn_hidden_units']
        fnn_init = learner_params['fnn_params']['fnn_init']
        nb_fnn_layers = len(nb_fnn_hidden_units)
        fnn_hid_act = learner_params['fnn_params']['fnn_hid_act']
        fnn_batchnorm_flag = learner_params['fnn_params']['fnn_batchnorm_flag']
        if nb_fnn_layers > 0:  # no need to read FNN parameters if there is no FNN layer (nb_fnn_flayers = 0).
            fnn_type = learner_params['fnn_params']['fnn_type']
            if fnn_type == 'Dense':
                fnn_type = Dense
            elif fnn_type == 'MaxoutDense':
                fnn_type = MaxoutDense
            else:
                raise ('unknown FNN type!')
            maxout_pool_size = learner_params['fnn_params']['maxout_pool_size']
            if fnn_type == Dense:
                maxout_pool_size = None

        # Read RNN layer parameters.
        nb_rnn_hidden_units = learner_params['rnn_params']['nb_rnn_hidden_units']
        nb_rnn_layers = len(nb_rnn_hidden_units)
        nb_rnn_proj_hidden_units = learner_params['rnn_params']['nb_rnn_proj_hidden_units']
        nb_rnn_proj_layers = len(nb_rnn_proj_hidden_units)
        if nb_rnn_layers > 0:  # no need to read RNN parameters if there is no RNN layer (nb_rnn_layers = 0).
            rnn_dropout_U = learner_params['rnn_params']['rnn_dropout_U']
            rnn_dropout_W = learner_params['rnn_params']['rnn_dropout_W']
            rnn_hid_act = learner_params['rnn_params']['rnn_hid_act']
            rnn_projection_downsampling = learner_params['rnn_params']['rnn_projection_downsampling']
            rnn_type = learner_params['rnn_params']['rnn_type']
            if rnn_type == 'GRU':
                rnn_type = GRU
            elif rnn_type == 'LSTM':
                rnn_type = LSTM
            else:
                raise ('unknown RNN type!')
            statefulness_flag = learner_params['rnn_params']['statefulness_flag']

        # Read CNN layer parameters.
        nb_conv_filters = learner_params['conv_params']['nb_conv_filters']
        nb_conv_layers = len(nb_conv_filters)
        # if nb_conv_layers > 0: # no need to read CNN parameters if there is no CNN layer.
        conv_act = learner_params['conv_params']['conv_act']
        conv_bord_mode = learner_params['conv_params']['conv_border_mode']
        conv_init = learner_params['conv_params']['conv_init']
        conv_stride = tuple(learner_params['conv_params']['conv_stride'])
        nb_conv_freq = learner_params['conv_params']['nb_conv_freq']
        nb_conv_time = learner_params['conv_params']['nb_conv_time']
        nb_conv_pool_freq = learner_params['conv_params']['nb_conv_pool_freq']
        nb_conv_pool_time = learner_params['conv_params']['nb_conv_pool_time']
        pool_stride_freq = learner_params['conv_params']['pool_stride_freq']
        pool_stride_time = learner_params['conv_params']['pool_stride_time']
        rnn_all_scan_flag = learner_params['conv_params'][
            'rnn_all_scan_flag']  # this is actually only used when conv layer is present (for now).
        batchnorm_flag = learner_params['conv_params']['batchnorm_flag']
        batchnorm_axis = learner_params['conv_params']['batchnorm_axis']

        nb_input_freq = input_shape
        nb_input_time = learner_params['input_data']['subdivs']
        nb_classes = learner_params['output_data']['nb_classes']
        nb_channels = 1

        b_reg = None
        b_constr = None
        w_constr = None

        if len(dropout_rate) == 1:  # If dropout rate is set to single value, expand the list for all layers
            #  (except RNN layers).
            dropout_rate = [dropout_rate[0]] * (nb_rnn_proj_layers + nb_conv_layers + nb_fnn_layers)
        assert len(dropout_rate) == nb_rnn_proj_layers + nb_conv_layers + nb_fnn_layers

        if len(nb_conv_freq) == 1:
            nb_conv_freq = [nb_conv_freq[0]] * nb_conv_layers
        assert len(nb_conv_freq) == nb_conv_layers

        if len(nb_conv_time) == 1:
            nb_conv_time = [nb_conv_time[0]] * nb_conv_layers
        assert len(nb_conv_time) == nb_conv_layers

        if len(nb_conv_pool_time) == 1:
            nb_conv_pool_time = [nb_conv_pool_time[0]] * nb_conv_layers
        assert len(nb_conv_pool_time) == nb_conv_layers

        if len(pool_stride_freq) == 0:
            pool_stride_freq = nb_conv_pool_freq
        elif len(pool_stride_freq) == 1:
            pool_stride_freq = [pool_stride_freq[0]] * nb_conv_layers

        if len(pool_stride_time) == 0:
            pool_stride_time = nb_conv_pool_time
        elif len(pool_stride_time) == 1:
            pool_stride_time = [pool_stride_time[0]] * nb_conv_layers

        if nb_conv_layers != 0:
            inputs = Input(shape=(nb_channels, nb_input_time, nb_input_freq))
        else:
            inputs = Input(shape=(1, nb_input_time, nb_input_freq))

        for i in range(nb_conv_layers):  # CNN layers loop
            if l1_weight_norm == 0. and l2_weight_norm == 0.:
                weight_reg = None
            else:
                weight_reg = L1L2(l1_weight_norm, l2_weight_norm)
            if i > 0:
                model = Convolution2D(filters=nb_conv_filters[i], kernel_size=(nb_conv_time[i], nb_conv_freq[i]),
                                      strides=conv_stride,
                                      padding=conv_bord_mode, data_format="channels_first",
                                      kernel_initializer=conv_init,
                                      kernel_regularizer=weight_reg, bias_regularizer=b_reg, kernel_constraint=w_constr,
                                      bias_constraint=b_constr)(model)
            else:
                model = Convolution2D(filters=nb_conv_filters[i], kernel_size=(nb_conv_time[i], nb_conv_freq[i]),
                                      strides=conv_stride,
                                      padding=conv_bord_mode, data_format="channels_first",
                                      kernel_initializer=conv_init,
                                      kernel_regularizer=weight_reg, bias_regularizer=b_reg, kernel_constraint=w_constr,
                                      bias_constraint=b_constr)(inputs)
            if batchnorm_flag:
                model = BatchNormalization(axis=batchnorm_axis)(model)
            if 'LeakyReLU' in conv_act:
                model = LeakyReLU()(model)
            else:
                model = Activation(conv_act)(model)
            if dropout_flag:
                model = Dropout(dropout_rate[i])(model)
            model = MaxPooling2D(pool_size=(1, nb_conv_pool_freq[i]),
                                 strides=(1, pool_stride_freq[i]), padding='valid', data_format='channels_first')(model)
            model = MaxPooling2D(pool_size=(nb_conv_pool_time[i], 1),
                                 strides=(pool_stride_time[i], 1), padding='same', data_format='channels_first')(model)

            nb_channels = nb_conv_filters[i]  # number of filters become the number of channels for the next CNN layer.
            if conv_bord_mode == 'valid':  # 'valid' convolution decreases the CNN output length by nb_conv_x -1.
                nb_input_time = int(
                    (nb_input_time - nb_conv_time[i] + conv_stride[0]) / nb_conv_pool_time[i] / conv_stride[0])
                nb_input_freq = int(
                    (nb_input_freq - nb_conv_freq[i] + conv_stride[1]) / nb_conv_pool_freq[i] / conv_stride[1])
            elif conv_bord_mode == 'same':
                nb_input_time = int(nb_input_time / conv_stride[0])
                nb_input_freq = int(nb_input_freq / nb_conv_pool_freq[i] / conv_stride[1])

            self.logger.debug(nb_input_time)
            self.logger.debug(nb_input_freq)
            self.logger.debug('\n')

        if nb_conv_layers > 0:  # Reshape CNN outputs for RNN or FNN layers.
            if rnn_all_scan_flag:  # Condense the outputs for each timestep and repeat with the number of timesteps.
                model = Flatten()(model)
                model = RepeatVector(nb_input_time)(model)
                nb_input_freq = nb_input_time * nb_channels * nb_input_freq
            else:
                # move the timesteps to the first axis and the number of channels to the second axis.
                model = Permute((2, 1, 3))(model)
                nb_input_freq = nb_channels * nb_input_freq
                # condense the last two axes (number of channels * outputs per channel).
                model = Reshape((nb_input_time, nb_input_freq))(model)
        for i in range(nb_rnn_layers):  # RNN layers loop.
            if statefulness_flag:
                if i == 0:
                    batch_inp_shape = (batch_size, nb_input_time, nb_input_freq)
                else:
                    if rnn_projection_downsampling:
                        batch_inp_shape = (batch_size, nb_input_time, nb_rnn_proj_hidden_units[i - 1])
                    else:
                        batch_inp_shape = (batch_size, nb_input_time, nb_rnn_hidden_units[i - 1])
                if i == 0 and nb_conv_layers == 0:
                    model = Reshape((nb_input_time, nb_input_freq))(inputs)

                model = rnn_type(units=nb_rnn_hidden_units[i], batch_input_shape=batch_inp_shape,
                                 activation=rnn_hid_act,
                                 return_sequences=True, stateful=statefulness_flag, dropout=rnn_dropout_W,
                                 recurrent_dropout=rnn_dropout_U, implementation=2)(model)

            else:
                if i == 0:
                    inp_shape = (nb_input_time, nb_input_freq)
                else:
                    if rnn_projection_downsampling:
                        inp_shape = (nb_input_time, nb_rnn_proj_hidden_units[i - 1])
                    else:
                        inp_shape = (nb_input_time, nb_rnn_hidden_units[i - 1])
                if i == 0 and nb_conv_layers == 0:
                    model = Reshape((nb_input_time, nb_input_freq))(inputs)

                model = rnn_type(units=nb_rnn_hidden_units[i], input_shape=inp_shape,
                                 activation=rnn_hid_act, return_sequences=True, stateful=statefulness_flag,
                                 dropout=rnn_dropout_W, recurrent_dropout=rnn_dropout_U, implementation=2)(model)

            if rnn_projection_downsampling:
                model = TimeDistributed(Dense(input_dim=nb_rnn_hidden_units[i], units=nb_rnn_proj_hidden_units[i],
                                              kernel_initializer=fnn_init))(model)
                if batchnorm_flag:
                    model = BatchNormalization(axis=-1)(model)
                model = Activation(fnn_hid_act)(model)
                if dropout_flag:
                    model = Dropout(dropout_rate[i + nb_conv_layers])(model)

        for i in range(nb_fnn_layers):  # FNN layers loop.
            if fnn_type == MaxoutDense:
                if i == 0 and nb_conv_layers + nb_rnn_layers == 0:  # First layer of the model as FNN
                    model = Reshape((nb_input_time, nb_input_freq))(inputs)
                model = TimeDistributed(fnn_type(units=nb_fnn_hidden_units[i], \
                                                 kernel_initializer=fnn_init, nb_feature=maxout_pool_size))(model)
            else:
                if i == 0 and nb_conv_layers + nb_rnn_layers == 0:  # First layer of the model as FNN
                    model = Reshape((nb_input_time, nb_input_freq))(inputs)
                model = TimeDistributed(fnn_type(units=nb_fnn_hidden_units[i], kernel_initializer=fnn_init))(model)

            if fnn_batchnorm_flag:
                model = BatchNormalization(axis=-1)(model)
            model = Activation(fnn_hid_act)(model)

            if dropout_flag:
                model = Dropout(dropout_rate[i + nb_conv_layers + nb_rnn_proj_layers])(model)

        if not learner_params['general']['temporal_max_pool']:
            # Add a TimeDistributed layer as an output layer.
            if learner_params['general']['last_maxout']:
                model = TimeDistributed(MaxoutDense(nb_classes, nb_feature=maxout_pool_size))(model)
            else:
                model = TimeDistributed(Dense(units=nb_classes, kernel_initializer=fnn_init, activation=output_act))(model)
            final_model = Model(inputs=inputs, outputs=model)
        else:
            if not learner_params['general']['long_short_branch']:
                model = MaxPooling1D(pool_size=learner_params['input_data']['subdivs'])(model)
                model = Flatten()(model)
                model = Dense(units=nb_classes, kernel_initializer=fnn_init, activation=output_act)(model)
                final_model = Model(inputs=inputs, outputs=model)
            else:
                model_long = MaxPooling1D(pool_size=learner_params['input_data']['subdivs'])(model)
                model_long = Flatten()(model_long)
                model_long = Dense(units=nb_classes, kernel_initializer=fnn_init, activation=output_act)(model_long)
                model_short = TimeDistributed(
                    Dense(units=nb_classes, kernel_initializer=fnn_init, activation=output_act))(model)
                final_model = Model(inputs=inputs, outputs=[model_short, model_long])

        self.model = final_model
        self.logger.debug(self.params)
        self.model.summary()
        self.compile_model()

    def compile_model(self):
        from keras.optimizers import SGD, RMSprop, Adagrad, Adam, Adadelta
        optimizer = self.learner_params['general']['optimizer']
        loss_func = self.learner_params['general']['loss']
        l_rate = self.learner_params['general']['learning_rate']
        if optimizer == 'sgd':
            optimizer = SGD(lr=l_rate)
        elif optimizer == 'rmsprop':
            optimizer = RMSprop(lr=l_rate)
        elif optimizer == 'adagrad':
            optimizer = Adagrad(lr=l_rate)
        elif optimizer == 'adam':
            optimizer = Adam(lr=l_rate)
        elif optimizer == 'adadelta':
            optimizer = Adadelta(lr=l_rate)
        else:
            pass
        self.model.compile(optimizer=optimizer, loss=loss_func, metrics=self.learner_params.get_path('model.metrics'))








class CustomAppCore(BinarySoundEventAppCore):
    def __init__(self, *args, **kwargs):
        kwargs['Learners'] = {
            'multifunctionalDNN_seq': MultifunctionalDeepLearningSequential,
        }
        super(CustomAppCore, self).__init__(*args, **kwargs)


def main(argv):
    numpy.random.seed(123456)  # let's make randomization predictable

    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            DCASE 2017
            Task 2: Rare Sound Event Detection
            A custom learner where you can create FNNs, CNNs, RNNs and CRNNs of any size and depth
            Code used in the paper:
            Convolutional Recurrent Neural Networks for Rare Sound Event Detection (Emre Cakir, Tuomas Virtanen)
            http://www.cs.tut.fi/sgn/arg/dcase2017/documents/challenge_technical_reports/DCASE2017_Cakir_104.pdf

            ---------------------------------------------
                Tampere University of Technology / Audio Research Group
                Author:  Emre Cakir ( emre.cakir@tut.fi )


        '''))

    # Setup argument handling
    parser.add_argument('-m', '--mode',
                        choices=('dev', 'challenge'),
                        default=None,
                        help="Selector for system mode",
                        required=False,
                        dest='mode',
                        type=str)

    parser.add_argument('-p', '--parameters',
                        help='parameter file override',
                        dest='parameter_override',
                        required=False,
                        metavar='FILE',
                        type=argument_file_exists)

    parser.add_argument('-s', '--parameter_set',
                        help='Parameter set id',
                        dest='parameter_set',
                        required=False,
                        type=str)

    parser.add_argument("-n", "--node",
                        help="Node mode",
                        dest="node_mode",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_sets",
                        help="List of available parameter sets",
                        dest="show_set_list",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_datasets",
                        help="List of available datasets",
                        dest="show_dataset_list",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_parameters",
                        help="Show parameters",
                        dest="show_parameters",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_eval",
                        help="Show evaluated setups",
                        dest="show_eval",
                        action='store_true',
                        required=False)

    parser.add_argument("-o", "--overwrite",
                        help="Overwrite mode",
                        dest="overwrite",
                        action='store_true',
                        required=False)

    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)

    # Parse arguments
    args = parser.parse_args()

    # Load default parameters from a file
    default_parameters_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               os.path.splitext(os.path.basename(__file__))[0] + '.defaults.yaml')
    if args.parameter_set:
        parameters_sets = args.parameter_set.split(',')
    else:
        parameters_sets = [None]

    for parameter_set in parameters_sets:
        # Initialize ParameterContainer
        params = ParameterContainer(project_base=os.path.dirname(os.path.realpath(__file__)),
                                    path_structure={
                                        'feature_extractor': [
                                            'dataset',
                                            'feature_extractor.parameters.*'
                                        ],
                                        'feature_normalizer': [
                                            'dataset',
                                            'feature_extractor.parameters.*'
                                        ],
                                        'learner': [
                                            'dataset',
                                            'feature_extractor',
                                            'feature_normalizer',
                                            'feature_aggregator',
                                            'learner'
                                        ],
                                        'recognizer': [
                                            'dataset',
                                            'feature_extractor',
                                            'feature_normalizer',
                                            'feature_aggregator',
                                            'learner',
                                            'recognizer'
                                        ],
                                    }
                                    )

        # Load default parameters from a file
        params.load(filename=default_parameters_filename)

        if args.parameter_override:
            # Override parameters from a file
            params.override(override=args.parameter_override)

        if parameter_set:
            # Override active_set
            params['active_set'] = parameter_set

        # Process parameters
        params.process()

        # Force overwrite
        if args.overwrite:
            params['general']['overwrite'] = True

        # Override dataset mode from arguments
        if args.mode == 'dev':
            # Set dataset to development
            params['dataset']['method'] = 'development'

            # Process dataset again, move correct parameters from dataset_parameters
            params.process_method_parameters(section='dataset')

        elif args.mode == 'challenge':
            # Set dataset to training set for challenge
            params['dataset']['method'] = 'challenge_train'
            params['general']['challenge_submission_mode'] = True
            # Process dataset again, move correct parameters from dataset_parameters
            params.process_method_parameters(section='dataset')

        if args.node_mode:
            params['general']['log_system_progress'] = True
            params['general']['print_system_progress'] = False

        # Force ascii progress bar under Windows console
        if platform.system() == 'Windows':
            params['general']['use_ascii_progress_bar'] = True

        # Setup logging
        setup_logging(parameter_container=params['logging'])

        app = CustomAppCore(name='DCASE 2017::Rare Sound Event Detection / Custom Multifunctional Deep Learning',
                             params=params,
                             system_desc=params.get('description'),
                             system_parameter_set_id=params.get('active_set'),
                             setup_label='Development setup',
                             log_system_progress=params.get_path('general.log_system_progress'),
                             show_progress_in_console=params.get_path('general.print_system_progress'),
                             use_ascii_progress_bar=params.get_path('general.use_ascii_progress_bar')
                             )

        # Show parameter set list and exit
        if args.show_set_list:
            params_ = ParameterContainer(
                project_base=os.path.dirname(os.path.realpath(__file__))
            ).load(filename=default_parameters_filename)

            if args.parameter_override:
                # Override parameters from a file
                params_.override(override=args.parameter_override)
            if 'sets' in params_:
                app.show_parameter_set_list(set_list=params_['sets'])

            return

        # Show dataset list and exit
        if args.show_dataset_list:
            app.show_dataset_list()
            return

        # Show system parameters
        if params.get_path('general.log_system_parameters') or args.show_parameters:
            app.show_parameters()

        # Show evaluated systems
        if args.show_eval:
            app.show_eval()
            return

        # Initialize application
        # ==================================================
        if params['flow']['initialize']:
            app.initialize()

        # Extract features for all audio files in the dataset
        # ==================================================
        if params['flow']['extract_features']:
            app.feature_extraction()

        # Prepare feature normalizers
        # ==================================================
        if params['flow']['feature_normalizer']:
            app.feature_normalization()

        # System training
        # ==================================================
        if params['flow']['train_system']:
            app.system_training()

        # System evaluation
        if not args.mode or args.mode == 'dev':

            # System testing
            # ==================================================
            if params['flow']['test_system']:
                app.system_testing()

            # System evaluation
            # ==================================================
            if params['flow']['evaluate_system']:
                app.system_evaluation()

        # System evaluation in challenge mode
        elif args.mode == 'challenge':
            # Set dataset to testing set for challenge
            params['dataset']['method'] = 'challenge_test'

            # Process dataset again, move correct parameters from dataset_parameters
            params.process_method_parameters('dataset')

            if params['general']['challenge_submission_mode']:
                # If in submission mode, save results in separate folder for easier access
                params['path']['recognizer'] = params.get_path('path.recognizer_challenge_output')

            challenge_app = CustomAppCore(name='DCASE 2017::Rare Sound Event Detection / Custom Multifunctional Deep Learning',
                                           params=params,
                                           system_desc=params.get('description'),
                                           system_parameter_set_id=params.get('active_set'),
                                           setup_label='Evaluation setup',
                                           log_system_progress=params.get_path('general.log_system_progress'),
                                           show_progress_in_console=params.get_path('general.print_system_progress'),
                                           use_ascii_progress_bar=params.get_path('general.use_ascii_progress_bar')
                                           )
            # Initialize application
            if params['flow']['initialize']:
                challenge_app.initialize()

            # Extract features for all audio files in the dataset
            if params['flow']['extract_features']:
                challenge_app.feature_extraction()

            # System testing
            if params['flow']['test_system']:
                if params['general']['challenge_submission_mode']:
                    params['general']['overwrite'] = True

                challenge_app.system_testing()

                if params['general']['challenge_submission_mode']:
                    challenge_app.ui.line(" ")
                    challenge_app.ui.line("Results for the challenge are stored at [" + params.get_path(
                        'path.recognizer_challenge_output') + "]")
                    challenge_app.ui.line(" ")

            # System evaluation if not in challenge submission mode
            if params['flow']['evaluate_system']:
                challenge_app.system_evaluation()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
