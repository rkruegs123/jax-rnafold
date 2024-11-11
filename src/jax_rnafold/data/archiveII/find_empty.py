import sys

for filename in sys.argv[1:]:
    with open(filename) as f:
        lines = [x for x in f]
    if all([x.split()[4]=="0" for x in lines[1:]]):
        print filename

