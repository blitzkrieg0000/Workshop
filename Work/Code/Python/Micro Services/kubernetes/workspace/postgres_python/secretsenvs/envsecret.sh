#!/bin/bash
kubectl create secret generic postgressecret --from-env-file=postgressecret.txt