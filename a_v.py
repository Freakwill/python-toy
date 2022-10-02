
import pathlib
import subprocess
import imageio
from PIL import Image


def video2mp3(video_path):
    """
    将视频转为音频
    :param video_path: 传入视频文件的路径
    :return:
    """
    if isinstance(video_path, str):
        video_path = pathlib.Path(video_path)
    outfile_name = video_path.with_suffix('.mp3')
    subprocess.call('ffmpeg -i ' + video_path
                    + ' -f mp3 ' + outfile_name, shell=True)

def video_add_mp3(video, mp3_file):
    """Add audio to a viedo
    
    Arguments:
        video {str|Path} -- path of video
        mp3_file {str} -- path of audio file
    """
    if isinstance(video, str):
        video = pathlib.Path(video)
    outfile_name = video.with_suffix('.add-mp3.mp4')
    subprocess.call('ffmpeg -i ' + str(video)
                    + ' -i ' + str(mp3_file) + ' -strict -2 -f mp4 '
                    + str(outfile_name), shell=True)


def compose_gif(path):
    """
        将静态图片转为gif动图
        :param path: 传入图片的目录的路径
        :return:
    """
    if isinstance(path, str): path = pathlib.Path(path)
    image_files = [p for p in path.iterdir() if p.suffix == ".png"]
    img_paths = sorted(image_files, key=lambda p: int(p.stem[3:]))
    img_paths = img_paths[:int(len(img_paths) / 3.6)]
    gif_images = [imageio.imread(path / img.with_suffix('out.png')) for img in img_paths]
    imageio.mimsave("test.gif", gif_images, fps=30)


def compress_png(path):
    """
        将gif动图转为每张静态图片
        :param path: 传入gif文件的路径
        :return:
    """
    if isinstance(path, str): path = pathlib.Path(path)
    image_files = [p for p in path.iterdir() if p.suffix == ".png"]
    for filename in image_files:
        with Image.open(filename) as im:
            width, height = im.size
            new_size = 150, int(new_width * height * 1.0 / width)
            resized_im = im.resize(new_size)
            resized_im.save(filename)


if __name__ == '__main__':
    # video2mp3(file_name='data-a.mp4')
    video_add_mp3(video='out.avi', mp3_file='paomo.mp3')
    # compose_gif(path='merged')
    # compress_png(path='merged')
