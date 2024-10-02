from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read(9585)


setup(
    name="ArabicTagger",  
    version="0.1.1",
    packages=find_packages(),
    install_requires=["tensorflow==2.10.0",\
                      "keras==2.10.0",\
                      "nltk",\
                       "bpemb", "numpy"],
    include_package_data=True,
    author="Eslam Tarek Farouk",
    author_email='cds.eslamtarek96337@alexu.edu.eg',
    description="A CRF, Encoder-Transformer layers, and BI-LSTM+CRF model were implemented in Keras with other modules.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EslamTarekFarouk/ArabicTagger",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
