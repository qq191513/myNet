import os
a = "1111111111\r\n"

def print_and_save_txt(str=None,filename=r'test_log.txt'):
    with open(filename, "a+") as log_writter:
        print(str)
        log_writter.write(str)





print_and_save_txt(a,r'test_log.txt')