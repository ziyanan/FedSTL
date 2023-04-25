import sys
import math
import torch
from random import randint
if (sys.version_info > (3, 0)):
    from functools import singledispatch
else:
    from singledispatch import singledispatch

from telex.stl import *
import operator as op



@singledispatch
def quantitativescore(stl, x, t):
    raise NotImplementedError("No quantitativescore for {} of class {}".format(stl, stl.__class__))

@quantitativescore.register(Globally)
def _(stl, x, t):
    (left, right) = stl.interval
    (maxtime, rangetime) = gettime(x, t+left, t+right)
    return  min(quantitativescore(stl.subformula, x, t1) for t1 in rangetime)

@quantitativescore.register(Future)
def _(stl, x, t):
    (left, right) = stl.interval
    (maxtime, rangetime) = gettime(x, t+left, t+right)
    return max(quantitativescore(stl.subformula, x, t1) for t1 in rangetime)

@quantitativescore.register(Until)
def _(stl, x, t):
    (left, right) = stl.interval
    left = float(left) 
    right = float(right) 
    if left>right:
        # print(left, right, "until q score")
        raise ValueError("Interval [{},{}] empty for {}".format(left, right, stl))    
    (maxtime, rangetime) = gettime(x, t+left, t+right)
    return max(quantitativescore(stl.right, x, t), max( min (quantitativescore(stl.right, x, t1), min (quantitativescore(stl.left, x, t2) for t2 in filter(lambda v: (v>= t) & (v<= t1) , rangetime) ) ) for t1 in rangetime) )

@quantitativescore.register(Or)
def _(stl, x, t):
    return max(quantitativescore(stl.left, x, t), quantitativescore(stl.right, x, t))

@quantitativescore.register(And)
def _(stl, x, t):
    return min(quantitativescore(stl.left, x, t), quantitativescore(stl.right, x, t))

@quantitativescore.register(Implies)
def _(stl, x, t):
    return max( (-1*  quantitativescore(stl.left, x, t) ), quantitativescore(stl.right, x, t) )

@quantitativescore.register(Not)
def _(stl, x, t):
    return -1 * quantitativescore(stl.subformula, x, t)

robusttable = { "<" : lambda x,y: y-x, "<=" : lambda x,y: y-x, ">" : lambda x,y: x-y , ">=": lambda x,y: x-y, "==" : lambda x,y: -abs(x,y) }

@quantitativescore.register(Constraint)
def _(stl, x, t):
    return robusttable[stl.relop](getval(stl.term, x, t), getval(stl.bound, x, t))

@quantitativescore.register(Atom)
def _(stl, x, t):
    if isinstance(x, dict):
        if x[stl.name][t]:
            return 1
        else:
            return 0
    else:
        if x[t]:
            return 1
        else:
            return 0





@singledispatch
def smartscore(stl, x, t):
    raise NotImplementedError("No smartscore for {} of class {}".format(stl, stl.__class__))

@smartscore.register(Globally)
def _(stl, x, t):
    (left, right) = stl.interval
    intervalwidth = right - left + 1
    (maxtime, rangetime) = gettime(x, t+left, t+right)
    return  2/(1 + math.exp(-0.01 * intervalwidth) ) * min(smartscore(stl.subformula, x, t1) for t1 in rangetime)

@smartscore.register(Future)
def _(stl, x, t):
    (left, right) = stl.interval
    intervalwidth = right - left + 1
    (maxtime, rangetime) = gettime(x, t+left, t+right)
    return  2/(1 + math.exp(0.01 * intervalwidth) ) * max(smartscore(stl.subformula, x, t1) for t1 in rangetime)

@smartscore.register(Until)
def _(stl, x, t):
    (left, right) = stl.interval
    left = float(left) 
    right = float(right) 
    if left>right:
        raise ValueError("Interval [{},{}] empty for {}".format(left, right, stl))    
    intervalwidth = right - left + 1
    (maxtime, rangetime) = gettime(x, t+left, t+right)
    return 2/(1 + math.exp(-0.01 * intervalwidth) ) * max(quantitativescore(stl.right, x, t), max( min (quantitativescore(stl.right, x, t1), min (quantitativescore(stl.left, x, t2) for t2 in filter(lambda v: (v>= t) & (v<= t1) , rangetime) ) ) for t1 in rangetime) )

@smartscore.register(Or)
def _(stl, x, t):
    return max(smartscore(stl.left, x, t), smartscore(stl.right, x, t))

@smartscore.register(And)
def _(stl, x, t):
    return min(smartscore(stl.left, x, t), smartscore(stl.right, x, t))

@smartscore.register(Implies)
def _(stl, x, t):
    return max( (-1*smartscore(stl.left, x, t) ), smartscore(stl.right, x, t) )

@smartscore.register(Not)
def _(stl, x, t):
    return -1 * smartscore(stl.subformula, x, t)

robusttable = { "<" : lambda x,y: y-x, "<=" : lambda x,y: y-x, ">" : lambda x,y: x-y , ">=": lambda x,y: x-y, "==" : lambda x,y: -abs(x,y) }

@smartscore.register(Constraint)
def _(stl, x, t):
    rawscore = robusttable[stl.relop](getval(stl.term, x, t), getval(stl.bound, x, t))
    return 1/(rawscore + torch.exp(-1*rawscore)) - torch.exp(-1*rawscore) 

@smartscore.register(Atom)
def _(stl, x, t):
    if isinstance(x, dict):
        if x[stl.name][t]:
            return torch.ones(1)
        else:
            return torch.zeros(1)
    else:
        if x[t]:
            return torch.ones(1)
        else:
            return torch.zeros(1)

def gettime(x, left, right):
    if isinstance(x, dict):
        ts = len(x[list(x.keys())[0]])
        maxtime = ts-1
        rangetime = [i for i in range(int(left), int(right)+1)]
        return maxtime, rangetime
    else:
        ts = len(x)
        maxtime = ts-1
        rangetime = [i for i in range(int(left), int(right)+1)]
        return maxtime, rangetime



@singledispatch
def getval(term, x, t):
    raise NotImplementedError("No getval for {} of class {}".format(stl, stl.__class__))

optable = { "<" : op.lt, ">" : op.gt, "<=" : op.le, ">=" : op.ge, "==": op.eq, "+" : op.add, "-" : op.sub, "*" : op.mul, "/" : op.truediv }
@getval.register(Expr)
def _(term, x, t):
    return optable[term.arithop](getval(term.left, x, t), getval(term.right, x, t))

@getval.register(Var)
def _(term, x, t):
    if isinstance(x, dict):
        return x[term.name][t]
    else:
        return x[t]

@getval.register(Constant)
def _(term, x, t):
    return term 
    
@getval.register(Param)
def _(term, x, t):
    raise NotImplementedError("No getval for parameter {}".format(term))
