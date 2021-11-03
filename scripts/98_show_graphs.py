from using_deepsulci.anatomist import Anatomist
from using_deepsulci.database import UDPDatabase
from dico_toolbox.database import BVDatabase


def main():
    db = BVDatabase("/neurospin/dico/data/bv_databases/human/pclean")
    udp_db = UDPDatabase(
        from_env="/neurospin/dico/bcagna/projects/using_deepsulci/scripts/env_basic_learning.json")

    hemi = "L"
    cname = 'pclean12A'

    cohort = udp_db.get_cohort(cname, hemi)
    sub = cohort.subjects[0].name

    whitemesh = db.get_from_template(
        "morphologist_mesh", type="white", subject=sub, hemi=hemi, analysis="default_analysis")[0]
    graph = udp_db.get_from_template("evaluation", train_cohort='pclean50A', model="unet3d_d00b01",
                                     run="01", subject=sub, hemi=hemi, extension="arg")[0]
    print(whitemesh)
    print(graph)
    ana = Anatomist()
    ana.show_labelled_graph(whitemesh, graph)


if __name__ == "__main__":
    main()
