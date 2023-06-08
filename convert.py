import imageio 
import cv2

video_path="video\Humanoid-v2\DDPG.mp4"
content=imageio.get_reader(video_path,"MP4")
frames = []
for j in range(500):
    try:
        frame = content.get_next_data()
        frame = cv2.resize(frame,(800,400))
        frames.append(frame)
    except IndexError:
        break
imageio.mimwrite(video_path.replace("mp4","gif"),frames,"GIF")