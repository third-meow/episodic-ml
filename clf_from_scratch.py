from gini_calc import *


training_data = [
        ['Green',  3, 'Apple'],
        ['Yellow', 3, 'Apple'],
        ['Red',    1, 'Grape'],
        ['Red',    1, 'Grape'],
        ['Yellow', 3, 'Lemon']]


class Question:

    def __init__(self):





def split():

    pass

def info_gain():
    pass




def find_best_split(data):
    best_so_far = [0, None]
    current_uncertainty = gini(data)
    n_feats = len(data[0]) - 1

    for col in range (n_feats):
        values = set([row[col] for row in data])

        for val in values:

            question = Question(col, val)

            true_data, false_data = split(data, question)

            if len(true_data) == 0 or len(false_data) == 0:
                continue

            gain = info_gain(true_data, false_data, current_uncertianty)

            if gain > best_do_far[0]:
                best_so_far = gain, question
    return best_so_far


def build_tree(data):

    gain, question = find_best_split(data)

    if gain == 0:
        #exit or something
        pass

    true_data, false_data = split(data, question)


    true_branch = build_tree(true_data)
    false_branch = build_tree(false_data)


    #return something
