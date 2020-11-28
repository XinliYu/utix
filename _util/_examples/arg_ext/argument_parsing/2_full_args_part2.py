# this script shows an example how to use `argx.get_parsed_args` for general argument setup, using the ArgInfo objects

import utix.arg_ext as argx

args = argx.get_parsed_args(argx.ArgInfo(full_name='para1_is_int', short_name='p', default_value=1),
                            argx.ArgInfo(full_name='para2_is_str', short_name='p', default_value='value'),
                            argx.ArgInfo(full_name='para3_is_list', default_value=[1, 2, 3, 4]))

# try `-p 2 -p1 3 -pil '[4,5,6,7]'` and `--para1_is_int 2 --para2_is_str 3 --para3_is_list '[4,5,6,7]'` again, and it should print out '2 3 [4, 5, 6, 7]'
# try `-p 2 -p1 3 -pil 5`
print(args.para1_is_int, args.para2_is_str, args.para3_is_list)
print(type(args.para1_is_int), type(args.para2_is_str), type(args.para3_is_list))
