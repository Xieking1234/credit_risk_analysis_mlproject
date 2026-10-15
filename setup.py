from setuptools import find_packages,setup
from typing import List

hyphen_e = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    returns list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements] #\n is removed cause it comes in req.txt file cause the packages are on lines one after other

        if hyphen_e in requirements:
            requirements.remove(hyphen_e) #to remove -e . in the requirements file at the end so it doenst come in setup.py for installation    
    return requirements        

setup(
    name = 'e2e_mlproject',
    version = '0.0.1',
    author = 'Ram',
    author_email='ramkruishna_llpl@hotmail.com',
    packages = find_packages(),
    install_requires=get_requirements('requirements.txt')
)