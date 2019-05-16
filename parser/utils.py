import requests
def readpretrain(filename):
    data = []
    with open(filename, "r") as r:
        while True:
            l = r.readline().strip()
            if l == "":
                break
            data.append(l.split())
    return data

def get_from_ix(w, to_ix, unk):
    if w in to_ix:
        return to_ix[w]
    assert unk != -1, "no unk supported"
    return unk

def getRequiredOutput(structs, tokens, sent):
    sentAdjusted = '+'.join(sent.split(' '))
    url = 'http://bollin.inf.ed.ac.uk:6123/?callback=jQuery111007715895978823618_1557984690708&sent={}&_=1557984690709'.format(sentAdjusted)
    r = requests.get(url)
    k = r.text.find('DRS(')
    p = k
    while r.text[p] != '\\':
        p += 1
    return r.text[k:p]