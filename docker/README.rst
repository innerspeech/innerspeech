Docker images
=============

We provide two docker images for the InnerSpeech project ``innerspeech-dev`` and ``innerspeech-core``.

``innerspeech-dev`` is a full capacity image that includes all the dependencies to run the InnerSpeech project, including the training and evaluation of the models.

``innerspeech-core`` is a lightweight image that includes only the dependencies to run the models and perform inference.

To build the images, you can use the provided Dockerfiles in the ``docker`` directory. You can also download the images from our Docker Hub repository.

For ``innerspeech-dev``:    

``docker pull innerspeech/innerspeech-dev``

``docker build -t innerspeech-dev ./docker/dev``

For ``innerspeech-core``:    

``docker pull innerspeech/innerspeech-core``   

``docker build -t innerspeech-core ./docker/core``

Github actions
--------------

We provide a Github action to build and push the images to the Docker Hub repository. The action is defined in the ``.github/workflows/build.yml`` file. 

The action is triggered on all Pull Requests and on all pushes to the ``main`` branch. The action will build the images and push them to the Docker Hub repository.