def FastModularExponentiation(b, e, m):
    itr=1
    rem=0
    tempe=e
    remlist=[]
    while(tempe>=1):
        tempb=b
        itr=1
        rem=0
        while(itr<=tempe):
            rem=tempb%m
            tempb=rem*rem
            itr=itr*2
        tempe=tempe-(itr/2)
        #print(tempe)
        remlist.append(rem)
    print(remlist)
    pro=1
    for x in remlist:
        pro=pro*x
        pro=pro%m
    return pro


