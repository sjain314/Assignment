import pandas as pd


def DataSet(dataSet):
    C = []
    for tid in dataSet:
        for item in tid:
            if not [item] in C:
                C.append([item])
    C.sort()
    return list(map(frozenset, C))


def DataScan(dataSet, CK, min_support, numItem, k=0):  # Accept candidate k itemsets and output frequent k itemsets
    ssCnt = {}
    for tid in dataSet:
        for can in CK:
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    retList = []
    supportData = {}
    for key in ssCnt:
        support = float(ssCnt[key] / numItem)
        if support >= min_support:
            retList.insert(0, key)

        # Scan D again and delete the infrequent k itemset. This improvement is to compress the transaction database and reduce the scanned data
        else:

            for tid in dataSet:
                if key == tid:
                    dataSet.remove(tid)

        supportData[key] = support

    R_List = []
    # if a single item i appears less than k times, i cannot appear in the frequent k+1 itemset. Items containing a single i should be deleted from the frequent K itemset and then linked
    # Improvement direction: compression candidate set CK
    if k > 1:
        ssCnt = {}
        for tid in retList:
            for key in tid:
                if key not in ssCnt:
                    ssCnt[key] = 1
                else:
                    ssCnt[key] += 1
        tids = []
        for tid in retList:
            for item in tid:
                if item in ssCnt.keys():
                    if ssCnt[item] < k:
                        tids.append(tid)
        R_List = list(set(retList) - set(tids))

    print(
        'Frequent itemsets before optimization' + str(retList) + '              ' + 'Optimized frequent itemsets' + str(
            R_List))
    return retList, supportData, R_List


def aprioriGen(LK, k, RK):  # Create candidate item set CK, where k is the number of elements in the output set

    if RK:
        LK = RK
    else:
        pass

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


def apriori(dataSet, min_support_1):
    C1 = DataSet(dataSet)
    D = set()
    for tid in dataSet:
        tid = frozenset(tid)
        D.add(tid)
    numItem = float(
        len(D))  # This is the only numItem to be calculated. Otherwise, the element of data list D will be deleted by scanD method, resulting in the change of numItem
    L1, supportData, R1 = DataScan(D, C1, min_support_1, numItem)
    L = [L1]
    R = [R1]
    k = 2
    while len(L[k - 2]) > 0:
        CK = aprioriGen(L[k - 2], k, R[k - 2])
        LK, supK, RK = DataScan(D, CK, min_support_1, numItem, k)
        supportData.update(supK)
        L.append(LK)
        R.append(RK)
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
