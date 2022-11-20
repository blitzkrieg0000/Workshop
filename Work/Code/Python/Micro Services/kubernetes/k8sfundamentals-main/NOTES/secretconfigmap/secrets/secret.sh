#!/bin/bash
kubectl create secret generic mysql-test-secret \
--from-file=MYSQL_ROOT_PASSWORD=../data/mysql_root_password.txt \
--from-file=MYSQL_USER=../data/mysql_user.txt \
--from-file=MYSQL_PASSWORD=../data/mysql_password.txt \
--from-file=MYSQL_DATABASE=../data/mysql_database.txt