from setuptools import setup, find_packages


setup(name='GCL',
      version='1.0.0',
      description='Joint Generative and Contrastive Learning for Unsupervised Person Re-identification',
      author='Hao Chen',
      author_email='hao.chen@inria.fr',
      url='https://github.com/chenhao2345/GCL',
      install_requires=[
          'numpy', 'torch==1.2.0', 'torchvision==0.4.0',
          'six', 'h5py', 'Pillow', 'scipy', 'tensorboard', 'opencv-python',
          'scikit-learn', 'metric-learn', 'faiss-gpu==1.6.3'],
      packages=find_packages(),
      keywords=[
          'Novel View Synthesis'
          'Contrastive Learning',
          'Unsupervised Person Re-identification'
      ])