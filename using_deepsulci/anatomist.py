import anatomist.api as ana
import os.path as op
import json

class Anatomist():
    def __init__(self) -> None:
        self.a = ana.Anatomist()

    def show_labelled_graph(self, mesh_f, graph_f, point_of_view='left_to_right', save_as=None,
                            hie_f="/casa/host/build/share/brainvisa-share-5.0/nomenclature/hierarchy/sulcal_root_colors.hie"):
        hie = self.a.loadObject(hie_f)

        mesh = self.a.loadObject(mesh_f)
        graph = self.a.loadObject(graph_f)

        # view an object
        win = self.a.createWindow("3D", geometry=[133, 93, 747, 516])
        win.addObjects([mesh, graph], add_graph_nodes=True)

        views = {
            'right_to_left': [0.5, -0.5, -0.5, 0.5],
            'left_to_right': [0.5, 0.5, 0.5, 0.5]
        }

        win.camera(view_quaternion=views[point_of_view])

        win.windowConfig(cursor_visibility=0)
        if save_as:
            win.snapshot(save_as)
            del win, mesh, graph


def main():
    point_of_view = 'left_to_right'
    hemi = 'L'
    sub = '001'

    db = "/host/home/bastien/projects/dico_toolbox/dico_toolbox_tests/test_data/database/mri-center/" + \
        sub + "/t1mri/default_acquisition/default_analysis"
    mesh_f = db + "/segmentation/mesh/" + sub + "_" + hemi + "white.gii"
    graph_f = db + "/folds/3.3/session1_manual/L001_session1_manual.arg"
    out_f = sub + '_snapshot.jpg'
    show_labelled_graph(mesh_f, graph_f, point_of_view, out_f)


if __name__ == "__main__":
    main()
