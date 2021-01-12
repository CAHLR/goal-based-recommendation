import argparse
import sys


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--target_course_id', type=int, default=0)

    parser.add_argument('--prereqs_path', type=str, default='../synthetic_data_samples/synthetic_prereqs_pairs.csv',
                        help='path to the official prerequisite pairs for evaluation')

    parser.add_argument('--course_id_path', type=str, default='../data_preprocess/course_id.pkl',
                        help='path to the generated course dictionary')

    parser.add_argument('--major_id_path', type=str, default='../data_preprocess/major_id.pkl',
                        help='path to the generated major dictionary')

    parser.add_argument('--grade_cutoff', type=str, default='A',
                        help='grade cutoff to separate grades into two categories - should be consistent with the value in training')

    parser.add_argument('--evaluated_model_path', type=str, default='../grade_prediction/models/LSTM1_50lr0.001drp0wd0clp0.pkl',
                        help='the path to the model saved from training for evaluation')

    parser.add_argument('--topN', type=int, default=10,
                        help='Top N results for evaluation')

    parser.add_argument('--results_path', type=str, default='results/',
                        help='path to the generated results folder to save the correctly predicted course pairs')


    return parser.parse_args()