__author__ = 'jwj'
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_enrollment', type=str, default='../synthetic_data_samples/synthetic_enrollment_data.csv',
                    help='path to input enrollment data')

    parser.add_argument('--input_major', type=str, default='../synthetic_data_samples/synthetic_major_data.csv',
                    help='path to input major data')

    parser.add_argument('--enrollment_threshold', type=int, default=20,
                        help='filter out courses with enrollments fewer than 20 times.')

    return parser.parse_args()