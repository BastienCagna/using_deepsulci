
def add_to_text_file(fname, s):
    f = open(fname, 'a')
    f.write(s)
    f.close()
