# x-map
Medical Diagnostics Using Deep Learning

For now, simple CNN integrated with Docker.


## To do:
- If necessary, set-up GPU avaliability for docker containers + integrate w. model
- Look at how to export singular images
- Docker vs-code development integration
- Wandb as visual set-up? (Are there issues with that?)
- Data-augmentation, if not enough training images or just to explore
- what CNN to choose?
- How do we test?
- Attention mechanisms(heatmaps of diagnostics)
- Load data

## How to run:
1. Install Docker Desktop-client (https://www.docker.com/get-started)
2. Open docker desktop
3. Navigate to project-location
4. run ```docker build -t image-name-here .``` in cmd to build the docker environment for the application and,
5. ```docker run image-name-here``` to run the application
