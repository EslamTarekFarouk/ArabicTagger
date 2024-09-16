from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="ArabicTagger",  
    version="0.1.2b2",
    packages=find_packages(),
    install_requires=["tensorflow==2.10.0",\
                      "keras==2.10.0",\
                      "nltk",\
                       "bpemb", "numpy"],
    include_package_data=True,
    author="Eslam Tarek Farouk",
    author_email='cds.eslamtarek96337@alexu.edu.eg',
    description="A CRF layer and BI-LSTM+CRF model implemented in Keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EslamTarekFarouk/ArabicTagger",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
