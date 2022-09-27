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
> 1. Make Sure eduVPN connected and the GitLab Container Registory owns docker image ```gitlab.ldv.ei.tum.de:5005/ami2022/group06```. If it does not exist, please push the previous local built image to the Gitlab Container Registory with the docker image name ```gitlab.ldv.ei.tum.de:5005/ami2022/group06``` 
> 2. Due to network fluctuations, the Webpage may be stuck, you can go back and refresh the page, and you can use it again
> 3. Active learning part generally takes about 5-8 minutes for retainning the model. If the network is stuck or the kubernets performance is limited, there is a possibility of failure. At this time, please re-upload the image for model correction.
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