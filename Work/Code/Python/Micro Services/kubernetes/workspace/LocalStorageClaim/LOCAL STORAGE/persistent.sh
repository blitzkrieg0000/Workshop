#!/bin/bash
kubectl apply -f storageClass_noprovisioner.yml
kubectl apply -f persistentVolume.yml
#kubectl apply -f persistentVolumeClaim.yml