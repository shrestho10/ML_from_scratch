def func(x):
    #anything
    return 0


def make_repeater(func, n):

    def repeater(x):
        result = x            
        i = 0
        while i < n:         
            result = func(result)
            i += 1
        return result  


    return repeater          


myfunc=make_repeater(func,4)
print(myfunc(2))