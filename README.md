~~~ bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python
pip install yt-dlp
yt-dlp -f "best[ext=mp4][height<=720]" "https://www.youtube.com/watch?v=_1VIxOLyaUQ" -o video.mp4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip uninstall opencv-python
pip install opencv-python-headless
pip uninstall opencv-python-headless opencv-python opencv-contrib-python
pip install opencv-python
pip install ultralytics
~~~
# semaforos-inteligentes
