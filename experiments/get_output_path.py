import os


def get_output_path(i, data, model_name):
    return os.path.join('results', 'result_' + str(i), '{}_{}'.format(data, model_name))
