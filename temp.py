lines = ['-DOCSTART- -X- -X- O\n',
         '\n',
         'SOCCER NN B-NP O\n',
         '- : O O\n',
         'JAPAN NNP B-NP B-LOC\n',
         'GET VB B-VP O\n',
         'LUCKY NNP B-NP O\n',
         'WIN NNP I-NP O\n',
         ', , O O\n',
         'CHINA NNP B-NP B-PER\n']

for line in lines:
    print(len(line.split()))


