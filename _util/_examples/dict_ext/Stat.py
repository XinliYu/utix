from _util.stat_util import Stat

stat = Stat()
stat.count(stat_name='buy', increase=3)
stat.count(stat_name='buy', increase=5)
print(stat.get_all_stats())

stat.average(stat_name='price', value=33.6)
stat.average(stat_name='price', value=10.1)
stat.average(stat_name='price', value=20.0)
print(stat.get_all_stats())

stat.aggregate(stat_name='discount', value=0.2, agg_func=max)
stat.aggregate(stat_name='discount', value=0.1, agg_func=max)
stat.aggregate(stat_name='discount', value=0.3, agg_func=max)
stat.aggregate(stat_name='discount', value=0.4, agg_func=max)
print(stat.get_all_stats())
