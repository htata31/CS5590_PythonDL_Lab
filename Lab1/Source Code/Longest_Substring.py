def Substring(_str):
    temp_str= ""
    dict1= {}
    for j in range(len(_str)):
        for i in range(j,len(_str)):
            if not(_str[i] in temp_str):
                temp_str+=_str[i]
            else :
                dict1[temp_str]= len(temp_str)
                temp_str =''
                break
    max_val = max(dict1.values())
    list1=[]
    for key , val in dict1.items():
        if(max_val == val):
            list1.append((key,val))
    print(list1)


if __name__ == '__main__':
    Substring("pwwkew")