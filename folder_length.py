import os

# path = "/workspace/reforumComfyXL/reforumxl/outputs/trippyferrfluid"
path = "/workspace/reforumComfyXL/reforumxl/outputs/trippyferrfluid"

pathlist = [f for f in os.listdir(path) if f.endswith(".png")]
vidlist = [f for f in os.listdir(path) if f.endswith(".mp4")]

if __name__ == "__main__":
    print(f"Total Images: {len(pathlist)}")
    print(f"Total Video: {len(vidlist)}")