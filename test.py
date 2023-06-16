class state(object):
    def __init__(self):
        pass

    def e(self):
        return 1

if __name__ == '__main__':
    a = state()
    b = state()
    m = {a: 1, b: 1}
    print(list(m.keys()))
    for key in m.keys():
        print(key.e() )