import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Autoencoders", # Replace with your own username
    version="0.0.1",
    author="Nate Groebner",
    author_email="groe0029@umn.edu",
    description="Autoencoders for seismic clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ngroebner/Autoencoders",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)