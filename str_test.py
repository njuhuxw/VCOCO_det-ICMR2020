a = "1_2_3_4"
b = "1_2"


def str_opera(str):
    str_list = str.split("_")
    if len(str_list) > 1:
        str_list.pop(-1)
    result = "_".join(str_list)
    print(result)
    print(str_list)


if __name__ == '__main__':
    str_opera(a)
    str_opera(b)
