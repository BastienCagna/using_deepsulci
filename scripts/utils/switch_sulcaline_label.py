import sys
import nibabel.gifti as ng


def switch(gii_f, old_label, new_label):
    gii = ng.read(gii_f)
    tex = gii.darrays[0].data
    tex[tex == old_label] = new_label
    gii.darrays[0].data = tex
    ng.write(gii, gii_f)


if __name__ == "__main__":
    switch(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
