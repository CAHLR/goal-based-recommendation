import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default='../data_preprocess/stu_sem_major_grade_condense.pkl',
                        help='path to the preprocessed data for training')

    parser.add_argument('--course_id_path', type=str, default='../data_preprocess/course_id.pkl',
                        help='path to the generated course dictionary')

    parser.add_argument('--major_id_path', type=str, default='../data_preprocess/major_id.pkl',
                        help='path to the generated major dictionary')

    parser.add_argument('--model_path', type=str, default='models/',
                        help='path to the generated model and loss records')

    parser.add_argument('--grade_cutoff', type=str, default='B',
                        help='grade cutoff to separate grades into two categories')

    parser.add_argument('--model_type', type=str, default='LSTM',
                        help='model type: LSTM, GRU, or RNN')

    parser.add_argument('--drop_out', type=float, default=0,
                        help='dropout rate')

    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay rate')

    parser.add_argument('--clip_gradient', type=float, default=0,
                        help='clip gradient threshold')

    parser.add_argument('--nb_lstm_units', type=float, default=50,
                        help='number of nodes in hidden layer(s)')

    parser.add_argument('--nb_lstm_layers', type=float, default=1,
                        help='number of hidden layers')

    parser.add_argument('--batchsize', type=float, default=32,
                        help='batch size')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')

    parser.add_argument('--evaluated_semester', type=float, default=15,
                        help='the order position of the semester for validation/evaluation, which depends on the input data.'
                             'In this data, 15 refers to 2019 Fall as the semester for validation, 18 refers to 2020 Fall for evaluation')

    return parser.parse_args()