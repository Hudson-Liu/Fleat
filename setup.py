"""Fast Learning Rate Tuner"""

from setuptools import setup, find_packages

# Allows PyPi website to access README.md
with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name="Fleat",
    version="0.9.0",
    description="A learning rate tuner that predicts learning rates without running the model.",
    license="MIT License",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Hudson Liu",
    author_email="hudsonliu0@gmail.com",
    url="https://hudson-liu.github.io/fleat",
    #download_url = [insert github release link here]
    packages=["lr_predictor", "preprocessor"],
    keywords=["keras", "machine learning", "tensorflow", "image classification", "deep learning", "hyperparameters", "neural networks", "fleat"],
    classifiers=[
       "Development Status :: 4 - Beta",
       "Intended Audience :: Developers",
       "Intended Audience :: Education",
       "Programming Language :: Python :: 3",
       "License :: OSI Approved :: MIT License",
       "Operating System :: OS Independent"
    ],
    install_requires = [
        "opencv-python",
        "keras"
    ],
    
    python_requires=">=3.6",
)