from setuptools import setup, find_packages

setup(
    name="visdrone_video_database",
    version="0.1.0",
    description="VisDrone Video Database with Frame Segment Trees",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pymongo>=4.0.0",
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
        "pillow>=8.0.0",
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "tqdm>=4.62.0",
        "yolov5>=7.0.0",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "visdrone-video-db=app.main:main",
        ],
    },
)