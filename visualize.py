#!/usr/bin/env python3
# Developed by Zhen Luo, Junyi Ma, and Zijie Zhou
# This file is covered by the LICENSE file in the root of the project PCPNet:
# https://github.com/Blurryface0814/PCPNet
# Brief: Visualize script for range-image-based point cloud prediction
import os
import argparse

from pcpnet.utils.visualization import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser("./visualize.py")
    parser.add_argument(
        "--path", "-p", type=str, default=None, help="Path to point clouds"
    )
    parser.add_argument(
        "--sequence", "-s", type=str, default="08", help="Sequence to visualize"
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame")
    parser.add_argument("--end", type=int, default=None, help="End frame")
    parser.add_argument("--capture", "-c", action="store_true", help="Capture frames")
    args, unparsed = parser.parse_known_args()

    gt_path = os.path.join(args.path, args.sequence, "gt/")
    end = last_file(gt_path) if not args.end else args.end
    start = first_file(gt_path) if not args.start else args.start
    assert end > start
    print("\nRendering scans [{s},{e}] from:{d}\n".format(s=start, e=end, d=gt_path))

    path_to_car_model = "car_model/bus_ply.ply"
    vis = Visualization(
        path=args.path,
        sequence=args.sequence,
        start=start,
        end=end,
        capture=args.capture,
        path_to_car_model=path_to_car_model,
    )
    vis.set_render_options(
        mesh_show_wireframe=False,
        mesh_show_back_face=False,
        show_coordinate_frame=False,
    )
    vis.run()
