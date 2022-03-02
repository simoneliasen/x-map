#FROM: Parent Image
FROM ubuntu:latest 

#RUN: Commands ran when the image is buil
RUN apt update 
RUN apt install python3 -y
RUN apt install python3-pip -y

#Copy required libraries from requirements.txt
#Needs to be done sepperately, due to some cache issues with docker
COPY requirements.txt /usr/app/src/requirements.txt

#Select working directory
WORKDIR /usr/app/src

#RUN: Commands ran when the image is built
RUN pip3 install -r requirements.txt

#COPY everything from local directory to docker image 
COPY . .

#Command to run after image has been created
CMD ["python3", "main.py"]
