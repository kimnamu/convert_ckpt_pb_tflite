import tensorflow as tf
import argparse

def print_pb():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="assign the name of input pb file")
    parser.add_argument(
        "--output",
        type=str,
        default="./summary_pb.txt",
        help="assign the name of output file"
    )

    flags, unparsed = parser.parse_known_args()
    pb_file = flags.input
    output = flags.output

    graph = tf.Graph()
    graphDef = tf.compat.v1.GraphDef()
    with open(pb_file, "rb") as graphFile:
        graphDef.ParseFromString(graphFile.read())

    # read pb into graph_def
    with tf.io.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # print operations
    with open(output, 'w+') as f:
        for op in graph.get_operations():
            f.write(op.name)
            f.write('\t')
            f.write(op.type)
            f.write('\t[')
            if len(op.values()) == 0:
                continue
            if not str(op.values()[0].shape)=="<unknown>":
                for s in op.values()[0].shape:
                    f.write(str(s))
                    f.write(' ')
            f.write(']\n\n')

if __name__ == "__main__":
    print_pb()
