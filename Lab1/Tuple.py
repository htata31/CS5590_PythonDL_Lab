from collections import defaultdict

def tuple():
    list1 = [( 'John', ('Physics', 80)) , ('Daniel', ('Science', 90)), ('John', ('Science', 95)), ('Mark',('Maths', 100)), ('Daniel', ('History', 75)), ('Mark', ('Social', 95))]
    dict1 = defaultdict(list)
    for key , val in list1:
        dict1[key].append(val)
    print(dict1)



if __name__ == '__main__':
    tuple()