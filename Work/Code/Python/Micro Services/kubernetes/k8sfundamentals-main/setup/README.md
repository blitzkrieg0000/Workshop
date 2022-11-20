# Kurulum

**Kubernetes Kurulum** konusuyla ilgili dosyalara buradan erişebilirsiniz.

## kubeadm kurulum

https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/

**0:** sanal makine oluşturma

```
$ sudo multipass launch --name master -c 2 -m 2G -d 10G
$ sudo multipass launch --name node1 -c 2 -m 2G -d 10G
```

**1:** Iptables bridged traffic ayarı

```
$ sudo cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
br_netfilter
EOF
```

```
$ sudo cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-ip6tables = 1
net.bridge.bridge-nf-call-iptables = 1
EOF
```

```
$ sudo sysctl --system
```

**2:** containerd kurulumu

```
$ sudo cat <<EOF | sudo tee /etc/modules-load.d/containerd.conf
overlay
br_netfilter
EOF
```

```
$ sudo modprobe overlay
$ sudo modprobe br_netfilter
```

```
$ sudo cat <<EOF | sudo tee /etc/sysctl.d/99-kubernetes-cri.conf
net.bridge.bridge-nf-call-iptables  = 1
net.ipv4.ip_forward                 = 1
net.bridge.bridge-nf-call-ip6tables = 1
EOF
```

```
$ sudo sysctl --system
```

```
$ sudo apt-get update
$ sudo apt-get upgrade -y
$ sudo apt-get install containerd -y
$ sudo mkdir -p /etc/containerd
$ sudo su -
$ sudo containerd config default | tee /etc/containerd/config.toml
$ exit
$ sudo systemctl restart containerd
```

**3:** kubeadm kurulumu

```
$ sudo apt-get update
$ sudo apt-get install -y apt-transport-https ca-certificates curl
$ sudo curl -fsSLo /usr/share/keyrings/kubernetes-archive-keyring.gpg https://packages.cloud.google.com/apt/doc/apt-key.gpg
$ echo "deb [signed-by=/usr/share/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list
$ sudo apt-get update
$ sudo apt-get install -y kubelet kubeadm kubectl
$ sudo apt-mark hold kubelet kubeadm kubectl
```

**4:** kubernetes cluster kurulumu

```
$ sudo kubeadm config images pull

$ sudo kubeadm init --pod-network-cidr=192.168.0.0/16 --apiserver-advertise-address=<ip> --control-plane-endpoint=<ip>

$ sudo kubeadm init --pod-network-cidr=192.168.1.0/24 --apiserver-advertise-address=192.168.1.100 --control-plane-endpoint=192.168.1.100
```

```
ZATEN kubectl init komutunun çıktısında bunları yapmamız gerekiyor
$ sudo mkdir -p $HOME/.kube
$ sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
$ sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

```
NETWORKING PLUGIN
$ kubectl create -f https://docs.projectcalico.org/manifests/tigera-operator.yaml
$ kubectl create -f https://docs.projectcalico.org/manifests/custom-resources.yaml
```
