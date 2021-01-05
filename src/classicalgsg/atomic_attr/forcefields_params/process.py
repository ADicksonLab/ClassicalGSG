import numpy as np


def processUFF():

    rfile = open('UFF_AC26.dat', 'r')
    wfile = open('UFF_AC.dat', 'w')
    for line in rfile.readlines():
        line = line.strip()
        words = line.split()
        s = "{:10}{:3}".format(words[2], words[0])
        s = s + '\n'
        wfile.write(s)

    wfile.close()


def processMMFF():
    rfile = open('mmffvdw.par', 'r')
    wfile = open('mmff.dat', 'w')
    for line in rfile.readlines():
        line = line.strip()
        if line[0].isdigit():
            words = line.split()
            atype = words[0]
            alpha = float(words[1])
            N = float(words[2])
            A = float(words[3])
            G = float(words[4])
            radius = A * np.power(alpha, 0.25)
            epsilon = ((181.16 * (G ** 2) * (alpha ** 2))
                       / (2 * np.sqrt(alpha/N))) * (1.0/np.power(radius, 6))

            s = "{:5}{:12}{:12}".format(atype, np.round(radius, 3),
                                        np.round(epsilon, 3))
            s = s + '\n'
            wfile.write(s)

    wfile.close()


def processGhemical():
    count = 0
    rfile = open('ghemical.prm', 'r')
    for line in rfile.readlines():
        line = line.strip()
        if line.startswith('vdw'):
            words = line.split()
            print(words)
            count = count + 1
    print(count)


if __name__ == '__main__':
    processMMFF()
