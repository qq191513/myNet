from functools import reduce

def str2float(s):
  return reduce(lambda x,y:x+int2dec(y),map(str2int,s.split('.')))
def char2num(s):
  return{'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}[s]
def str2int(s):
  return reduce(lambda x,y:x*10+y,map(char2num,s))
def intLen(i):
    return len('%d' % i)
def int2dec(i):
  return i/(10** intLen(i))
