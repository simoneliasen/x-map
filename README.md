# x-map 
![alt text](https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/google/313/x-ray_1fa7b.png)

**Medical Diagnostics Using Deep Learning**
- For now, simple CNN integrated with Docker.



## How to run:
1. [Install Docker Desktop-client](https://www.docker.com/get-started)
3. Open docker desktop
4. Navigate to project-location
5. Run ```docker build -t image-name-here .``` in cmd to build the docker environment for the application
  - Do this each time you want to update the contents of your docker container.
7. ```docker run image-name-here``` to run the application


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


## Debugging
- Run Container interactively: ```docker run -it image-name-here /bin/bash```
- Export files from container to local system: ```docker export container-name-here > imgtest.tar```
- Copy singular file from Docker to Localhost:
  1. type out ```docker ps``` to get the Contained ID
  2. ```docker cp container-id-here:/opt/app/app.log .``` copy the specified file, to the local working directory
- Copy from Local host to Docker (easier way would be to just build again) ```docker cp /host/local/path/file container-id-here:/file/path/in/container/file```  
