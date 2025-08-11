from moviepy.video.VideoClip import TextClip, ImageClip  
from moviepy.video.compositing import CompositeVideoClip
from moviepy import VideoFileClip
from pathlib import Path

script_dir = Path(__file__).parent
print(script_dir)
file_name = 'elastico_skill_1'
input_path = script_dir / "untrimmed_data" / f"{file_name}.mp4"
output_path = script_dir / "trimmed_data" / f"{file_name}2.mp4"
print(input_path)
start_time = 15.2
end_time = 16.8

start_end_time = True

if not input_path.exists():
    raise FileNotFoundError(f"File not found: {input_path}")

clip = VideoFileClip(str(input_path))
if start_end_time:
    clip = clip.subclipped(start_time, end_time)
else:
    clip = clip.subclipped(start_time)
    
clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")
