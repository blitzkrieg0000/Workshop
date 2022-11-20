#!/bin/bash
kubectl create secret generic mysql-test-secret -n test \
--from-file=MYSQL_ROOT_PASSWORD=mysql_root_password.txt \
--from-file=MYSQL_USER=mysql_user.txt \
--from-file=MYSQL_PASSWORD=mysql_password.txt \
--from-file=MYSQL_DATABASE=mysql_database.txt

kubectl create secret generic mysql-prod-secret -n production \
--from-file=MYSQL_ROOT_PASSWORD=mysql_root_password.txt \
--from-file=MYSQL_USER=mysql_user.txt \
--from-file=MYSQL_PASSWORD=mysql_password.txt \
--from-file=MYSQL_DATABASE=mysql_database.txt