from more_itertools import locate

def isDuplicatedID(mylist):
    duplist = []
    for i in set(mylist):
        if mylist.count(i) > 1:
            indexPosList = list(locate(mylist, lambda a: a == i))
            print(i, mylist.count(i), indexPosList)
            duplist.append((i, mylist.count(i), indexPosList))
    if len(duplist) > 0:
        fp = open("duplicateID" + ".txt", "w", encoding="utf-8")
        for v in duplist:
            fp.write(str(v) + "\n")
        fp.close()
        return True
    else:
        return  False