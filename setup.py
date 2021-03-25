from setuptools import setup, find_packages


setup(name='GCL',
      version='1.0.0',
      description='Joint Generative and Contrastive Learning for Unsupervised Person Re-identification',
      author='Hao Chen',
      author_email='hao.chen@inria.fr',
      url='https://github.com/chenhao2345/GCL',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss-gpu'],
      packages=find_packages(),
      keywords=[
          'Novel View Synthesis'
          'Contrastive Learning',
          'Unsupervised Person Re-identification'
      ])