import pandas as pd


def DataSet(dataSet):
    C = []
    for tid in dataSet:
        for item in tid:
            if not [item] in C:
                C.append([item])
    C.sort()
    return list(map(frozenset, C))


def DataScan(dataSet, CK, minSupport):  # Accept candidate k itemsets and output frequent k itemsets
    ssCnt = {}
    for tid in dataSet:
        for can in CK:
            if can.issubset(tid):
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItem = float(len(dataSet))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = float(ssCnt[key] / numItem)
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support  # Update support dictionary
    return retList, supportData
    retList = []
    lenLK = len(LK)
    for i in range(lenLK):
        for j in range(i + 1, lenLK):
            L1 = list(LK[i])[:k - 2]
            L2 = list(LK[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(LK[i] | LK[j])
    return retList


def aprioriGen(LK, k):  # Create candidate item set CK, where k is the number of elements in the output set

    retList = []
    lenLK = len(LK)
    for i in range(lenLK):
        for j in range(i + 1, lenLK):
            L1 = list(LK[i])[:k - 2]
            L2 = list(LK[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(LK[i] | LK[j])
    return retList


def apriori(dataSet, minSupport):
    C1 = DataSet(dataSet)
    D = set()
    for tid in dataSet:
        tid = frozenset(tid)
        D.add(tid)
    L1, supportData = DataScan(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):
        CK = aprioriGen(L[k - 2], k,)
        LK, supK = DataScan(D, CK, minSupport)
        supportData.update(supK)
        L.append(LK)
        k += 1
    L = [i for i in L if i]  # Delete empty list
    return L, supportData


name = 'menu_orders.txt'
minSupport = 0.2
minconfidence = 0.5


def createData(name):  # Preprocess the data and output the dataset

    D = pd.read_csv(name, header=None, index_col=False, names=['1', '2', '3'])

    D = D.fillna(0)
    D = D.values.tolist()
    for i in range(len(D)):
        D[i] = [j for j in D[i] if j != 0]
    return D


def calculate(dataset):  # Calculation of support and confidence by algorithm
    dataset, dic = apriori(dataset, minSupport)

    Rname = []
    Rsupport = []
    Rconfidence = []
    emptylist = []
    for i in range(len(dataset)):
        for AB in dataset[i]:
            for A in emptylist:
                if A.issubset(AB):
                    conf = dic.get(AB) / dic.get(AB - A)
                    if conf >= minconfidence:
                        Rname.append(str(AB - A) + '-->' + str(A))
                        Rconfidence.append(conf)
                        Rsupport.append(dic.get(AB))

            emptylist.append(AB)
    return Rname, Rsupport, Rconfidence


def outputdata(Rname, Rsupport, Rconfidence):
    data = {

        "Association rules": Rname,
        "Support": Rsupport,
        "Confidence": Rconfidence
    }
    df = pd.DataFrame(data,
                      columns=['Association rules', 'Support', 'Confidence'])

    return df


dataset = createData(name)
R1, R2, R3 = calculate(dataset)
df = outputdata(R1, R2, R3)
df.to_csv('Report.txt')
