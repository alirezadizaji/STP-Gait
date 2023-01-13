class EdgeType:
    DEFAULT = "DEFAULT"
    """ Edges are the same as OpenPose Skeleton graph, just replicating it in frames """

    INTER_FRAME_M1 = "INTER_FRAME_M1"
    """ DEFAULT + 1st mode inter-frame edges between frames. Please checkout `utils.edge_processing.generate_inter_frames_edge_index_mode1`."""

    INTER_FRAME_M2 = "INTER_FRAME_M2"
    """ DEFAULT + 2nd mode inter-frame edges between frames. Please checkout `utils.edge_processing.generate_inter_frames_edge_index_mode2`."""

    CUSTOM = "CUSTOM"
    """ Custom edge definition """
