from setuptools import setup, find_packages
hype='-e .'
def get_requirements(file_path="requirements.txt"):
    with open(file_path) as f:
        requirements = f.readlines()
        if hype in requirements:
            requirements.remove(hype)
    return [req.replace('/n','') for req in requirements]
setup(
    name="mlproject",
    version="0.1.0",
    author="Abhishek",
    author_email="abhishekvk18op@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)