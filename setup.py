import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

    name="neuromaps",

    version="alpha",

    author="Javier Rasero",

    author_email="jrasero.daparte@gmail.com",

    description="A package to generate predictive maps from neuroimaging 4D data",

    long_description=long_description,

    #long_description_content_type="text/markdown",

    url="https://github.com/jrasero",

    packages=setuptools.find_packages(),

    classifiers=[

        "Programming Language :: Python :: 3",

        "License :: OSI Approved :: MIT License",

    ],

    python_requires='>=3.6',

)