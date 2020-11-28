# this script shows an example how to use `argx.get_parsed_args` for general argument setup, using 2-tuples.

import utix.arg_ext as argx

# we can provide argument info tuples;
# here every tuple is a 2-tuple, 1) the first being the name in the format of `fullname/shortname`, or just the `fullname`, and 2) the second being the default value;
# NOTE if the 'shortname' is not specified, the default is to use the first letter of the 'parts' of the full name as the short name.
# NOTE if the duplicate short name is found, an incremental number will be appended to the end to solve the name conflict.
args = argx.get_parsed_args(('para1_is_int/p', 1), ('para2_is_str/p', 'value'), ('para3_is_list', [1, 2, 3, 4]))  # short name not specified, and the default short name is `pil` by connecting the first letter of 'parts' of the full name

# 1) without any argument, this will print out "1 value [1, 2, 3, 4]";
# 2) set arguments by short names `-p 2 -p1 3 -pil '[4,5,6,7]'` and this will print out "2 3 [4, 5, 6, 7]", where the '3' is of string type;
# 3) set arguments `--para1_is_int 2 --para2_is_str 3 --para3_is_list '[4,5,6,7]'` and this will print out "2 3 [4, 5, 6, 7]", where the '3' is of string type,
print(args.para1_is_int, args.para2_is_str, args.para3_is_list)
print(type(args.para1_is_int), type(args.para2_is_str), type(args.para3_is_list))
