# this script shows an example how to use `argx.get_parsed_args` for general argument setup, with converters

import utix.arg_ext as argx

args = argx.get_parsed_args(argx.ArgInfo(full_name='para1_is_int', short_name='p', default_value=1),
                            argx.ArgInfo(full_name='para2_is_str', short_name='p', default_value='value', converter=lambda x: '_' + x.upper()),
                            argx.ArgInfo(full_name='para3_is_list', default_value=[1, 2, 3, 4], converter=lambda x: x ** 2),
                            argx.ArgInfo(full_name='para4_is_dict', default_value={'a': 1, 'b': 2}, converter=lambda k, v: (k, k + str(v))))

# 1) without any argument, this will print out "1 _VALUE [1, 4, 9, 16] {'a': 'a1', 'b': 'b2'}";
# 2) try `-p 2 -p1 3 -pil '[4,5,6,7]' -pid "{'a':2, 'b':3}"` and `--para1_is_int 2 --para2_is_str 3 --para3_is_list '[4,5,6,7]' --para3_is_dict "{'a':2, 'b':3}"` again,
#       and it should print out '2 _3 [16, 25, 36, 49] {'a': 'a2', 'b': 'b3'}'
print(args.para1_is_int, args.para2_is_str, args.para3_is_list, args.para4_is_dict)
print(type(args.para1_is_int), type(args.para2_is_str), type(args.para3_is_list), type(args.para4_is_dict))
