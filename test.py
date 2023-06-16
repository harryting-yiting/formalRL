import  sympy as sp
class state(object):
    def __init__(self):
        pass

    def e(self):
        return 1

def test1(aa):


    aa = [3]
    print((aa))
def test2(aa):
    aa.append(4)


if __name__ == '__main__':
    a = state()
    b = state()
    m = {a: 1, b: 1}
    print(list(m.keys()))
    for key in m.keys():
        print(key.e() )

    print(set() is None)

    bb = [1,3]
    test1(bb)
    print(bb)
    # [1, 3]

    test2(bb)
    print(bb)
    # [1,3,4]

    a = sp.symbols('a')
    print(len(set([1,2,3])))