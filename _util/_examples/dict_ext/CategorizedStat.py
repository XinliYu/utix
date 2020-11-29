from utix._util.stat_util import CategorizedStat

stat = CategorizedStat()
stat.count(category='music', stat_name='buy', value=[3, 5])
stat.count(category='video', stat_name='buy', value=[2, 1])
print(stat.get_stats(name='buy'))
stat.count(category='music', stat_name='buy', value=[1, 1])
stat.count(category='video', stat_name='buy', value=[0, -1])
print(stat.get_stats(name='buy'))

stat = CategorizedStat()
stat.count(category='music', stat_name='buy', value=3)
stat.count(category='video', stat_name='buy', value=5)
print(stat.get_stats(name='buy'))

stat.average(category='music', stat_name='price', value=33.6)
stat.average(category='video', stat_name='price', value=10.1)
stat.average(category='video', stat_name='price', value=20.0)
print(stat.get_stats(name='price'))

stat.aggregate(category='music', stat_name='discount', value=0.2, agg_func=max)
stat.aggregate(category='video', stat_name='discount', value=0.1, agg_func=max)
stat.aggregate(category='video', stat_name='discount', value=0.3, agg_func=max)
stat.aggregate(category='video', stat_name='discount', value=0.4, agg_func=max)

print(stat.get_stats(name='discount'))
