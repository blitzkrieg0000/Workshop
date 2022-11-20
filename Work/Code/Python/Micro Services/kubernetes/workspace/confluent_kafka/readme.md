# LOCAL PROVISIONER OLUŞTURMA
**NOT** : *Bu Provisioner sayesinde statefulsetlerimiz local olarak pvc ve pv üretebilecekler*
***
# 1-) NFS SUNUCUSU KURULUMU
```
$ sudo systemctl status nfs-server
$ sudo apt install nfs-kernel-server nfs-common portmap
$ sudo start nfs-server
$ sudo mkdir -p /srv/nfs/mydata 
$ sudo chmod -R 777 nfs/ # for simple use but not advised
$ sudo vi /etc/exports
    stdout: /srv/nfs/mydata  *(rw,sync,no_subtree_check,no_root_squash,insecure)
$ sudo exportfs -rv
    stdout: "exporting *:/srv/nfs/mydata"
$ showmount -e
    stdout: "/srv/nfs/mydata  *"

#FARKLI BİR MAKİNEDE ÇALIŞIYORSAK VEYA MINIKUBE KULLANIYORSAK
$ sudo mount -t nfs 192.168.1.100:/srv/nfs/mydata /mnt
```
***
# 2-) NFS KUBERNETES CLIENT HELM İLE KURULUMU
*Manuel Kurunca Ek Ayarlamalar Gerekebiliyor. O yüzden helm anında ayağa kaldırıyor.*
```
helm repo add nfs-subdir-external-provisioner https://kubernetes-sigs.github.io/nfs-subdir-external-provisioner/

helm install nfs-subdir-external-provisioner nfs-subdir-external-provisioner/nfs-subdir-external-provisioner \
    --set nfs.server=x.x.x.x \
    --set nfs.path=/exported/path
```
***
# 3-)CONFLUENT KAFKA CLUSTER
**TODO** : *Readiness Probeler tanımlanacak( depends_on ) böylece podların birbirini beklemesi sağlanacak*
```
$ ./env.sh #ConfigMapleri Yükler (ENVIRONMENT VARIABLES)
$ kubectl apply -f confluent-kafka-kubernetes.yml
```