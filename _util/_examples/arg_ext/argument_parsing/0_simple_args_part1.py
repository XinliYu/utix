# this script shows an example how to use `argx.get_parsed_args` to quickly setup terminal arguments

import utilx.arg_ext as argx

# by simply specifying the default values, it tells the function there should be three terminal arguments `para1`, `para2` and `para3`,
# and it hints the function that `para1` is of type `int`, `para2` is of type `str`, and `para3` is of type `list`
args = argx.get_parsed_args(default_para1=1, default_para2='value', default_para3=[1, 2, 3, 4])

# 1) without any argument, this will print out "1 value [1, 2, 3, 4]";
# 2) set arguments `--para1 2 --para2 3 --para3 '[4,5,6,7]'` and this will print out "2 3 [4, 5, 6, 7]", where the '3' is of string type;
# 3) set arguments `--para1 2 --para2 3 --para3 5`  and this will print out "2 3 [5]", where the '3' is of string type, and the '5' is turned into a list;
# 4) the short names for these arguments are automatically generated, and they are 'p', 'p1', 'p2', try `-p 2 -p1 3 -p2 5`; we'll see more about short names later.
print(args.para1, args.para2, args.para3)
print(type(args.para1), type(args.para2), type(args.para3))
