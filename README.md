# x-map
Medical Diagnostics Using Deep Learning

For now, simple CNN integrated with Docker.


## How to run:
1. [Install Docker Desktop-client](https://www.docker.com/get-started)
3. Open docker desktop
4. Navigate to project-location
5. run ```docker build -t image-name-here .``` in cmd to build the docker environment for the application and,
6. ```docker run image-name-here``` to run the application


## To do:
- [ ] Figure out training, testing, validation allocation and metrics
- [ ] Use correct training + test data
- [ ] Setup GPU avalability in container (also needs to be configured with model)
- [ ] Higher ease of use with continious development
- [ ] How to export singular files from docker container (e.g. images)
- [ ] Should/can Wandb be used as the only data visualization tool?
- [ ] What CNN to use?
- [ ] Data-augmentation to get more data
- [ ] Attention-mechanism to get heatmaps of diagnostics on image
- [ ] Hyperparameter tuning (probably with bayesian search)
