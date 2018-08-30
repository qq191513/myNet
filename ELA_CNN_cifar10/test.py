import collections
#其中field_names 有多种表达方式，如下
# ggg=collections.namedtuple('name age sex')
ddd=collections.namedtuple('student',['name','age','sex'])
hhh=collections.namedtuple('student','name,age,sex')

spark=ddd(name='sunYang',age=20,sex='male')
print(type(spark))
print("spark's name is %s" % spark.name)
print("%s is %d years old %s" % spark)

# 显示结果如下：
# student(name='sunYang', age=20, sex='male')
# spark's name is sunYang
# sunYang is 20 years old male