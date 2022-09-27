## Dockerized WEB Application

## How to run the image of our project locally:
>Prerequest: Docker, eduVPN
1. Connect to eduVPN and Open Docker
2. Open local Terminal and run the command: ```docker login gitlab.ldv.ei.tum.de:5005 ```
3. Run the command: ```docker run --rm -d  -p 8888:8888/tcp gitlab.ldv.ei.tum.de:5005/ami2022/group06:latest ```
    (The purpose of this step is to pull image from gitlab and run our container)
4. Open the browser: http://127.0.0.1:8888
    
> If all goes well, you will be able to see our page
Notes:

1. Check docker is installed with ``` docker -v ``` and VPN is connected to **tum.eduvpn.lrz.de**
2. Check Internet Status and Firewall Setting, Make sure Port 8888 is not occupied or forbidden
3. First time executing step 3, it normally require some time, please be patient :D 
   
### How to run the image of our project by Dcoker File locally

Git Clone the Web Folder from GitLab master branch and run follow commands will get image "gitlab.ldv.ei.tum.de:5005/ami2022/group06" and container "gitlab.ldv.ei.tum.de:5005/ami2022/group06" for test
```
docker build --tag gitlab.ldv.ei.tum.de:5005/ami2022/group06 .
docker run -d -p 8888:8888 gitlab.ldv.ei.tum.de:5005/ami2022/group06
```

### How to run the image of our project by Docker Compose File locally

Git Clone the Web Folder from GitLab master branch and run follow command will get image "gitlab.ldv.ei.tum.de:5005/ami2022/group06" and container "ami_group06_container", the image tag name and container name will be setted in docker-compose.yml file
```
docker-compose build
docker-compose up
```

## How to depoly our docker image in Kubernetes:

1. First ckeck if it successfully connect and then check if it exists running pods and service in kubernetes. When it have service **service-group06** and deployment **deployment-group06**, you can directly access the Website: http://10.195.8.77:30625/
``` 
kubectl get nodes
kubectl get deployment
kubectl get service
```

2. When the above service and delopyment don't exist, you can use this commmand to devoply the Docker image in Kubernetes
```
kubectl apply -f group06_deployment_service.yaml
```
3. Check out status of pod. At beginnig, it is ContainerCreating, after a few minutes it becomes running.
```   
kubectl get pods
kubectl describe pods
```
4. You can directly access the Website: http://10.195.8.77:30625/ , or you can use below command to map the port to a local port and access the 127.0.0.1:8888, filnally you can enjoy website
```
kubectl port-forward service/service-group06 8888:8888
```
5. After using, you can delete the service and deployment, secret to terminate the pods
```
kubectl delete svc service-group06
kubectl delete deployments deployment-group06
kubectl delete secret imgcred-group06
```
> Tips
> 1. Make Sure eduVPN connected and the GitLab Container Registory owns docker image ```gitlab.ldv.ei.tum.de:5005/ami2022/group06``` . If it does not exist, please push the previous local built image to the Gitlab Container Registory with the docker image name ```gitlab.ldv.ei.tum.de:5005/ami2022/group06``` 
> 2. Due to network fluctuations, the Webpage may be stuck, you can go back and refresh the page, and you can use it again
> 3. Active learning part generally takes about 5-8 minutes for retrainning the model. If the network is stuck or the kubernets performance is limited, there is a possibility of failure. At this time, please re-upload the image for model correction.
> 4. Don't worry about information "Timeout occurred" or "port aborted", the service is still running and you can still get the result

#### Configuration of private SAMBA share

1. Check out the status of PersistentVolume and PersistentVolumeClaim

```
kubectl get pv
kubectl get pvc
```

2. If the PersistentVolume **pvc-smb-group06** and PersistentVolumeClaim **pvc-smb-group06** exists, We don't need to execute the following command in step 3 to step 5

3. Add the credential to access SAMBA share

```
kubectl apply -f group06_smbcreds.yaml
```
4. Create the PersistentVolume **pvc-smb-group06**

```
kubectl apply -f group06_samba_pv.yaml
```
5. Claim the PersistentVolume **pvc-smb-group06** and create a deployment for test
```
kubectl apply -f group06_pvc.yaml
```

6. Now in the server it has created the PersistentVolume **pvc-smb-group06** and PersistentVolumeClaim **pvc-smb-group06**

> Tips
> 1. This part referes to the **samba_mount** folder in **ami-examples** repository
> 2. Normmaly we don't need execute this part commands. It has already successfully configured :-O 
> 3. Private SAMBA share Configuration isn't prerequisites to our Web deployment in Kubernetes.

## Introduction to WEB
There is page on our web that explain how to use it: http://10.195.8.77:30625/support. Here are some characteristics of our website  

* Classification

    1. Upload Image
    2. Model setting
    3. Detection
    4. Manuel Correction and Refine Model (Active Learning)
    5. Output the Final Report

* Final Report Download
  
    We can see all detection result from the selceted model in form of tabel and Using the button you can download the report as Json file with name "final_report.json". 

    ```
    {
    "annotations": [
        {
            "file_name": "7025372_236.jpg",
            "label": "Dent"
        },
        {
            "file_name": "7025381_34.jpg",
            "label": "Scratch"
        },
        {
            "file_name": "7025392_376.jpg",
            "label": "Rim"
        }
            ]
    }
    ```

* Image Label

    you can uplaod the unlabeled images and labeled by human. The unlabeled images will go into the labeled image tabel after labeled. And the information of labeled iamges can be downloaded as Json file with name "image_label.json". 
     ```
    {
    "labels": [
        {
            "file_name": "7025381_34_svZ1aXh.jpg",
            "label": "Dent"
        }
            ]
    }
    ```
