import numpy as np

def get_output(string):
    i = 0
    output = [chr(97 + int(string[0]))]

    for j in range(1, len(string)):
        if string[j] != string[j-1]:
            output.append(chr(97 + int(string[j])))
        else:
            continue
    output.reverse()
    output = ''.join(output)
    return output


def viterbi(observation, initial_prob, observation_prob, transition_prob_matrix):
    viterbi_matrix = np.zeros((26, len(observation)))
    viterbi_index = np.zeros((26, len(observation)))
    output_viterbi = []

    rows, columns = viterbi_matrix.shape

    for column in range(columns):
        if column == 0:
            for row in range(rows):
                viterbi_matrix[row, column] = np.log(initial_prob[row]) + np.log(observation_prob[row, observation[column]])
                viterbi_index[column, column] = -1
        else:
            for row in range(rows):
                prev_column = viterbi_matrix[:, column-1]
                temp_transition = np.log(transition_prob_matrix[:, row])
                trans_prod = np.add(prev_column, temp_transition)
                index = np.argmax(trans_prod)
                max_value = trans_prod[index]
                viterbi_matrix[row, column] = max_value + np.log(observation_prob[row, observation[column]])
                viterbi_index[row, column] = index

    max_index = -1
    for column in range(columns - 1, 0, -1):
        if column == columns - 1:
            max_index = np.argmax(viterbi_matrix[:, column])
            output_viterbi.append(max_index)
            max_index = viterbi_index[int(max_index), column]
        else:
            output_viterbi.append(max_index)
            # print("max ind : ", max_index)
            # print(column)
            max_index = viterbi_index[int(max_index), column]

    # print(viterbi_matrix[:, 1322:])

    return output_viterbi


'''read files'''
with open('initialStateDistribution.txt') as file:
    initial_prob = np.loadtxt(file, delimiter=',', dtype=float)
    # initial_prob = np.array(initial_prob, dtype=float)

with open('transitionProbMatrix.txt') as file:
    transition_prob_matrix = np.loadtxt(file, dtype=float, delimiter=',')

with open('observationProbMatrix.txt') as file:
    observation_prob_matrix = np.loadtxt(file, delimiter=',', dtype=float)

with open('observations_art.txt') as file:
    observations_art = np.loadtxt(file, delimiter=',', dtype=int)

with open('obsr_1_1.txt', encoding='latin') as file:
    # observations_test = np.loadtxt(file, delimiter=',', dtype=int)
    d = file.read().split(',')
    d = [int(i) for i in d]
    d = np.array(d)

result = viterbi(d, initial_prob, observation_prob_matrix, transition_prob_matrix)
# print(np.unique(result))
print(get_output(result))
