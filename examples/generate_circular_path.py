import json
import math
import os
import argparse
import numpy as np

def create_circular_path(r, h, num_frames=120, fps=30.0, default_fov=50.0):
    camera_path = []
    keyframes = []
    
    for i in range(num_frames):
        theta = 2.0 * math.pi * i / num_frames
        
        # Position
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        z = h
        pos = np.array([x, y, z])
        
        # Camera Z axis (points AWAY from origin, OpenGL convention)
        c_z = pos / np.linalg.norm(pos)
        
        # Up vector (World Z)
        up = np.array([0.0, 0.0, 1.0])
        
        # Camera X axis (points RIGHT)
        c_x = np.cross(up, c_z)
        c_x_norm = np.linalg.norm(c_x)
        if c_x_norm < 1e-6:
            # Handle the case where camera is exactly above/below origin
            c_x = np.array([1.0, 0.0, 0.0])
        else:
            c_x = c_x / c_x_norm
            
        # Camera Y axis (points UP in camera local space)
        c_y = np.cross(c_z, c_x)
        c_y = c_y / np.linalg.norm(c_y)
        
        # Create 4x4 matrix
        matrix = np.eye(4)
        matrix[:3, 0] = c_x
        matrix[:3, 1] = c_y
        matrix[:3, 2] = c_z
        matrix[:3, 3] = pos
        
        matrix_flat = matrix.flatten().tolist()
        
        frame = {
            "camera_to_world": matrix_flat,
            "fov": default_fov,
            "aspect": 1.3333333333333333
        }
        camera_path.append(frame)
        
        # Optionally add as keyframes
        keyframe = {
            "matrix": matrix_flat,
            "fov": default_fov,
            "aspect": 1.3333333333333333,
            "override_transition_enabled": False,
            "override_transition_sec": None
        }
        keyframes.append(keyframe)
        
    data = {
        "default_fov": default_fov,
        "default_transition_sec": 1.0,
        "keyframes": keyframes,
        "render_height": 960.0,
        "render_width": 1280.0,
        "fps": fps,
        "seconds": num_frames / fps,
        "is_cycle": True,
        "smoothness_value": 0.0,
        "camera_path": camera_path
    }
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Generate a circular camera path JSON file")
    parser.add_argument("--r", type=float, required=True, help="Radius of the circular path")
    parser.add_argument("--h", type=float, required=True, help="Height of the circular path")
    parser.add_argument("--frames", type=int, default=150, help="Number of frames")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    parser.add_argument("--out", type=str, default="circular_path.json", help="Output JSON filename")
    
    args = parser.parse_args()
    
    print(f"Generating camera path with radius r={args.r} and height h={args.h} (frames={args.frames})")
    
    data = create_circular_path(args.r, args.h, args.frames, args.fps)
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, args.out)
    
    with open(out_path, "w") as f:
        json.dump(data, f)
        
    print(f"Successfully saved camera path to: {out_path}")

if __name__ == "__main__":
    main()
