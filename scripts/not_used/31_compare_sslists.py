from soma import aims
from utils import sulci_list_from_graph
import sys


def list_new_items(old_list, new_list):
    new_items = []
    for new_item in new_list:
        if new_item not in old_list:
            new_items.append(new_item)
    return new_items


def main(graph_a_f, graph_b_f):
    ssl_a = sulci_list_from_graph(aims.read(graph_a_f))
    ssl_b = sulci_list_from_graph(aims.read(graph_b_f))

    new_in_b = list_new_items(ssl_a, ssl_b)
    new_in_a = list_new_items(ssl_b, ssl_a)

    print("Labels in both graphs:")
    for it in sorted(set(ssl_a + ssl_b)):
        if it in ssl_a and it in ssl_b:
            print(it)

    print("Labels in A but not in B")
    for it in new_in_a:
        print(it)

    print("Labels in B but not in A")
    for it in new_in_b:
        print(it)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
