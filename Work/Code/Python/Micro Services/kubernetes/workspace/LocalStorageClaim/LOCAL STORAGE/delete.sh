#!/bin/bash
kubectl delete pvc brokerdataclaim-brokerdeployment-0
kubectl delete pvc zookeeperdataclaim-zookeeperdeployment-0
kubectl delete pv local-pv
kubectl delete sc local-storage