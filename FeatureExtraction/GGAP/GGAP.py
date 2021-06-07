# amino = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
#          'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
import pandas as pd
import numpy as np
import readFasta
import re
import sys
import os
from collections import Counter
# pPath = os.path.split(os.path.realpath(__file__))[0]
# sys.path.append(pPath)

USAGE = """
USAGE:
	python GGAP.py input.fasta order <output>
	input.fasta:      the input protein sequence file in fasta format.
	order:            the out order, select from ['alphabetically', 'polarity', 'sideChainVolume' or userDefined] 
	output:           the encoding file, default: 'encodings.tsv'
"""


def GGAP(fastas, gap, ** kw):
    AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    header = ['#']
    patterns = []
    for aa1 in AA:
        for aa2 in AA:
            header.append(aa1 + aa2)
            patterns.append(aa1 + "[" + AA + "]" + "{" + str(gap) + "}" + aa2)
    encodings.append(header)
    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        length = len(sequence)
        denominator = length - gap - 1
        for pattern in patterns:
            fre = len(re.findall(pattern, sequence)) / denominator
            code.append(fre)
        encodings.append(code)
    return encodings, header


kw = {'path': r"H_train.txt", 'order': 'ACDEFGHIKLMNPQRSTVWY'}

if __name__ == "__main__":
    fastas1 = readFasta.readFasta(r"data.fasta")
    result, header = GGAP(fastas1, 3, **kw)
    data1 = np.matrix(result[1:])[:, 1:]
    data2 = np.matrix(result[1:])

    # data2 = np.matrix(result)
    data_GGAP = pd.DataFrame(data=data1, columns=header[1:])
    data_GGAP_name = pd.DataFrame(data=data2, columns=header)
    data_GGAP.to_csv('GGAP_data.csv')
    data_GGAP_name.to_csv('GGAP_data_name.csv')
