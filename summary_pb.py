import tensorflow as tf

def print_pb(pb_file):

    graph = tf.Graph()
    graphDef = tf.GraphDef()
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
    with open("pb_information.txt", 'w+') as f:
        for op in graph.get_operations():
            f.write(op.name)
            f.write('\t')
            f.write(op.type)
            f.write('\t[')
            if len(op.values()) == 0:
                continue
            print(op.values()[0].shape)
            print(type(op.values()[0].shape))
            if not str(op.values()[0].shape)=="<unknown>":
                for s in op.values()[0].shape:
                    f.write(str(s))
                    f.write(' ')
            f.write(']\n\n')

            
print_pb("./model/sample.pb")