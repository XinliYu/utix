# this script shows an example how to use `argx.get_parsed_args` to quickly setup terminal arguments

import utilx.arg_ext as argx

# if no default values are needed, we could just specify the names;
# NOTE that without default values, there is no way to infer the type of each argument, unless it can be recognized as list, a tuple, a set or a dictionary;
# all other arguments will be of string type;
args = argx.get_parsed_args('para1', 'para2', 'para3')

# 1) without any argument, this will print out an empty line, and the types are '<class 'str'> <class 'str'> <class 'str'>';
# 2) try arguments `--para1 2 --para2 3 --para3 '[4,5,6,7]'` and this will print out "2 3 [4, 5, 6, 7]", with types '<class 'int'> <class 'str'> <class 'list'>'.
# 3) try arguments `--para1 2 --para2 3 --para3 5` and `-p 2 --p1 3 --p2 5`, and it will print out "2 3 5" with types '<class 'str'> <class 'str'> <class 'str'>'.
print(args.para1, args.para2, args.para3)
print(type(args.para1), type(args.para2), type(args.para3))
