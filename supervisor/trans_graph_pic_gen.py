import os
import json
import cv2 

from .picture_generator import PictureGenerator

class TransGraphPictureGenerator(PictureGenerator):
    def run(self):
        if not self.session_path:
            return
        graphs_dir = os.path.join(self.session_path, "graphs")
        json_path = os.path.join(graphs_dir, "trans_graph_PDDL.json")
        if not os.path.exists(json_path):
            return

        try:
            with open(json_path, 'r') as f:
                graph_data = json.load(f)
            img = self.renderer.create_base_canvas("TRANSITION GRAPH", f"based on PDDL domain")
            self.renderer.draw_transition_graph(img, graph_data)

            out_dir = self.get_pictures_dir()
            if not out_dir:
                return
            out_path = os.path.join(out_dir, "trans_graph_PDDL.png")
            cv2.imwrite(out_path, img)
            print(f"[PictureGenerator] Saved Transition Graph image to {out_path}")
        except Exception as e:
            print(f"[PictureGenerator] Failed to generate TransGraph: {e}")